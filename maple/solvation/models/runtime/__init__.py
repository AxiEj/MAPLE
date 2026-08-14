"""Optional heavy runtime implementations for Route-2 model adapters.

Contract modules in :mod:`maple.solvation.models` must remain importable without
Torch or model runtimes.  Heavy implementations live here and are imported only
inside explicit runtime builders.
"""
