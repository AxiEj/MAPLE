"""Closed value-only copy of the frozen harmonic recurrence for compiler probes.

This research surface removes the lazy import from the hot function. Arithmetic
and normalization follow harmonic_torch_primitives._torch_real_harmonic_design;
no compiler result is treated as a derivative implementation or admitted PES.
"""

import math

import torch


def make_harmonic_value_kernel(lmax):
    phase = tuple((-1)**m * math.prod(range(1, 2*m, 2)) for m in range(lmax+1))
    normalization = tuple(
        math.sqrt((2*l+1)*math.exp(math.lgamma(l-abs(m)+1)-math.lgamma(l+abs(m)+1))
                  / (4.0*math.pi))
        for l in range(lmax+1) for m in range(-l,l+1)
    )

    def kernel(directions):
        x, y, z = directions.unbind(-1)
        values = {}
        xy = torch.complex(x, y)
        for m in range(lmax+1):
            if m == 0:
                cosine, sine = torch.ones_like(z), torch.zeros_like(z)
            else:
                sectoral = xy**m
                cosine, sine = phase[m]*sectoral.real, phase[m]*sectoral.imag
            c, s = {m:cosine}, {m:sine}
            if m < lmax:
                c[m+1], s[m+1] = (2*m+1)*z*cosine, (2*m+1)*z*sine
            for l in range(m+2,lmax+1):
                c[l] = ((2*l-1)*z*c[l-1] - (l+m-1)*c[l-2])/(l-m)
                s[l] = ((2*l-1)*z*s[l-1] - (l+m-1)*s[l-2])/(l-m)
            for l in range(m,lmax+1):
                values[l,m,"c"], values[l,m,"s"] = c[l], s[l]
        columns = []
        for l in range(lmax+1):
            for signed_m in range(-l,l+1):
                m = abs(signed_m)
                norm = normalization[l*l+l+signed_m]
                if signed_m < 0:
                    column = math.sqrt(2.0)*((-1)**m)*norm*values[l,m,"s"]
                elif signed_m == 0:
                    column = norm*values[l,0,"c"]
                else:
                    column = math.sqrt(2.0)*((-1)**m)*norm*values[l,m,"c"]
                columns.append(column)
        return torch.stack(columns,dim=-1)
    return kernel
