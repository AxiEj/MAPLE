"""Isolated full-resolution continuum controls; not a molecular PES benchmark."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import threading
import time
import traceback

import numpy as np
import psutil
import torch

from streamed_ddpcm import StreamedDDPCM, solve_diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--geometry", choices=("chain", "compact"), default="chain")
    parser.add_argument("--mode", choices=("actions", "solve"), default="actions")
    parser.add_argument(
        "--kernel",
        choices=("eager", "compiled-real", "fused-eager", "fused-compiled"),
        default="eager",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous probe outputs.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    n = args.atoms
    rng = np.random.default_rng(61001)
    if args.geometry == "chain":
        x = np.arange(n, dtype=float)
        positions = np.column_stack(
            (1.63 * x, 0.31 * np.sin(x * 1.37), 0.23 * np.cos(x * 0.79))
        )
    else:
        side = int(np.ceil(n ** (1 / 3)))
        ijk = np.array(list(np.ndindex(side, side, side)))[:n]
        positions = 2.7 * ijk + rng.normal(scale=0.07, size=(n, 3))
    source = rng.normal(scale=0.025, size=(n, 4))
    source[:, 0] -= source[:, 0].mean()
    source[:, 1:] *= 0.1
    r, s = torch.tensor(positions), torch.tensor(source)
    process = psutil.Process()
    peaks = {
        "rss_bytes": process.memory_info().rss,
        "uss_bytes": process.memory_full_info().uss,
    }
    done = threading.Event()
    started = time.perf_counter()

    def sample():
        while not done.wait(0.1):
            memory = process.memory_full_info()
            peaks["rss_bytes"] = max(peaks["rss_bytes"], memory.rss)
            peaks["uss_bytes"] = max(peaks["uss_bytes"], memory.uss)
            # Measurement only: the external cgroup supervisor enforces the
            # process-tree memory and wall budgets, including compiler children.

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    result = {
        "scope": "synthetic fixed-source continuum control only",
        "scientific_admitted": False,
        "complete": False,
        "arguments": vars_as_dict(args),
        "lmax": 15,
        "n_lebedev": 1202,
        "eta": 0.1,
        "dtype": "torch.float64",
        "geometry_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "prototype_sha256": hashlib.sha256(
            Path(__file__).with_name("streamed_ddpcm.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    from research_identity import research_python_hashes, verify_tracked_freeze

    result["research_sources_before"] = research_python_hashes()
    result["tracked_freeze_before"] = verify_tracked_freeze()["manifest_sha256"]
    snapshots = Path(__file__).parent / "source-snapshots"
    snapshots.mkdir(exist_ok=True)
    for name, digest in result["research_sources_before"].items():
        target = snapshots / (digest + "-" + name)
        if not target.exists():
            target.write_bytes(Path(__file__).with_name(name).read_bytes())
    result["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_threads": torch.get_num_threads(),
        "timing_scope": "cold process; first action includes any JIT; not a qualified speedup",
    }
    try:
        before = time.perf_counter()
        cls = StreamedDDPCM
        if args.kernel == "compiled-real":
            from compiled_value_operator import CompiledValueStreamedDDPCM

            cls = CompiledValueStreamedDDPCM
        elif args.kernel in {"fused-eager", "fused-compiled"}:
            from fused_value_operator import FusedValueStreamedDDPCM

            cls = FusedValueStreamedDDPCM
        kernel_options = (
            {"compile_kernels": args.kernel == "fused-compiled"}
            if args.kernel.startswith("fused-")
            else {}
        )
        op = cls(
            ("C",) * n,
            np.full(n, 1.7),
            r,
            dielectric=78.39,
            source_tile=args.tile,
            **kernel_options,
        )
        result["setup_seconds"] = time.perf_counter() - before
        result["storage"] = op.storage_report()
        result["topology"] = asdict(op.topology)
        if args.mode == "solve":
            result["solve"] = solve_diagnostic(op, s)
        else:
            x = torch.tensor(rng.normal(size=op.dimension))
            y = torch.tensor(rng.normal(size=op.dimension))
            rows = []
            for kind in ("L", "D"):
                before = time.perf_counter()
                ax = op.apply(kind, x)
                forward = time.perf_counter() - before
                before = time.perf_counter()
                aty = op.apply(kind, y, transpose=True)
                backward = time.perf_counter() - before
                lhs = float(y.dot(ax))
                rhs = float(x.dot(aty))
                error = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
                if not error < 1e-12:
                    raise AssertionError(f"{kind} transpose-pair error {error}")
                rows.append(
                    {
                        "operator": kind,
                        "forward_seconds": forward,
                        "transpose_seconds": backward,
                        "transpose_pair_relative_error": error,
                        "forward_norm": float(ax.norm()),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            result["actions"] = rows
        result["complete"] = True
    except Exception as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        done.set()
        thread.join()
        result["elapsed_seconds"] = time.perf_counter() - started
        result["sampled_process_peak"] = peaks
        result["ru_maxrss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        result["research_sources_after"] = research_python_hashes()
        result["tracked_freeze_after"] = verify_tracked_freeze()["manifest_sha256"]
        result["sources_unchanged"] = (
            result["research_sources_before"] == result["research_sources_after"]
            and result["tracked_freeze_before"] == result["tracked_freeze_after"]
        )
        if not result["sources_unchanged"]:
            result["complete"] = False
            result["failure"] = {
                "type": "SourceDrift",
                "message": "sources changed during the probe",
            }
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "peak_rss_GiB": result["ru_maxrss_bytes"] / 1024**3,
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
    return 0 if result["complete"] else 1


def vars_as_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


if __name__ == "__main__":
    raise SystemExit(main())
