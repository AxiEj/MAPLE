"""Isolated FP64 value-kernel experiment, never compile a PES/AD chain."""

import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

import torch

from maple.solvation.continuum.harmonic_torch_primitives import _torch_real_harmonic_design


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    out = Path(__file__).with_name("compiled-harmonics.json")
    if out.exists():
        raise FileExistsError("Do not overwrite earlier evidence.")
    generator = torch.Generator().manual_seed(1001)
    directions = torch.randn((8*1202,3),generator=generator,dtype=torch.float64)
    directions /= directions.norm(dim=1,keepdim=True)
    result = {"scope":"CPU harmonic values only; no derivative/PES qualification",
              "scientific_admitted":False,"complete":False,
              "dtype":"torch.float64","lmax":15,"point_count":len(directions),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        def kernel(x):
            return _torch_real_harmonic_design(x,lmax=15)
        expected = kernel(directions)
        compiled = torch.compile(kernel,fullgraph=True)
        start=time.perf_counter();actual=compiled(directions)
        result["compilation_and_first_seconds"]=time.perf_counter()-start
        error=float((actual-expected).abs().max())
        result["max_absolute_basis_error"]=error
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        timings={}
        for name,fn in (("eager",kernel),("compiled",compiled)):
            fn(directions)
            values=[]
            for _ in range(3):
                start=time.perf_counter();fn(directions);values.append(time.perf_counter()-start)
            timings[name]=values
        result["seconds"]=timings
        result["complete"]=True
    except Exception as error:
        result["failure"]={"type":type(error).__name__,"message":str(error),
                           "traceback":traceback.format_exc()}
    finally:
        result["process_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result,indent=2),flush=True)
    return 0 if result["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
