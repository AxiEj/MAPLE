"""Dependency-light sealed-descriptor loading for captured checkpoint bytes."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import fcntl
import os

_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008


@contextmanager
def sealed_checkpoint_descriptor(checkpoint_bytes: bytes):
    """Expose exact captured bytes through one sealed process-owned memfd."""

    if type(checkpoint_bytes) is not bytes or not checkpoint_bytes:
        raise TypeError("checkpoint_bytes must be nonempty exact bytes.")
    memfd_create = getattr(os, "memfd_create", None)
    if callable(memfd_create):
        descriptor = memfd_create(
            "maple-model-checkpoint",
            flags=_MFD_CLOEXEC | _MFD_ALLOW_SEALING,
        )
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        libc_memfd_create = getattr(libc, "memfd_create", None)
        if libc_memfd_create is None:
            raise RuntimeError(
                "secure checkpoint loading requires sealable Linux memfd."
            )
        libc_memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        libc_memfd_create.restype = ctypes.c_int
        descriptor = libc_memfd_create(
            b"maple-model-checkpoint",
            _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise RuntimeError(
                "secure checkpoint memfd creation failed: " + os.strerror(error_number)
            )
    try:
        view = memoryview(checkpoint_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("sealed checkpoint write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        captured = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            captured.extend(block)
        if bytes(captured) != checkpoint_bytes:
            raise RuntimeError("sealed checkpoint descriptor bytes changed.")
        required_seals = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
        try:
            fcntl.fcntl(descriptor, _F_ADD_SEALS, required_seals)
            actual_seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
        except OSError as exc:
            raise RuntimeError("checkpoint memfd sealing is unavailable.") from exc
        if actual_seals & required_seals != required_seals:
            raise RuntimeError("checkpoint memfd did not retain every required seal.")
        proc_path = f"/proc/self/fd/{descriptor}"
        try:
            descriptor_stat = os.fstat(descriptor)
            proc_stat = os.stat(proc_path)
        except OSError as exc:
            raise RuntimeError(
                "secure /proc/self/fd checkpoint access is unavailable."
            ) from exc
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            proc_stat.st_dev,
            proc_stat.st_ino,
        ):
            raise RuntimeError("checkpoint procfd does not identify the sealed memfd.")
        yield proc_path
    finally:
        os.close(descriptor)


__all__ = ["sealed_checkpoint_descriptor"]
