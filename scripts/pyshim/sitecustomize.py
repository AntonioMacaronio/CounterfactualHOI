"""Loaded automatically when scripts/pyshim is on PYTHONPATH (see scripts/serve_h3_ref2va.sh).

uv's and conda-forge's CPython builds are compiled against an old glibc sysroot, so they omit
os.memfd_create even though the host glibc (2.35) has it. SGLang's H3 reference-video path calls
os.memfd_create and only falls back on OSError, so a missing attribute crashes every Ref2VA
request. Provide it through ctypes.
"""

import os
import sys

if sys.platform.startswith("linux") and not hasattr(os, "memfd_create"):
    import ctypes

    _libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(_libc, "memfd_create"):
        _libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        _libc.memfd_create.restype = ctypes.c_int

        def memfd_create(name, flags=1):
            fd = _libc.memfd_create(os.fsencode(name), flags)
            if fd < 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            return fd

        os.memfd_create = memfd_create
        for _name, _value in (("MFD_CLOEXEC", 1), ("MFD_ALLOW_SEALING", 2)):
            if not hasattr(os, _name):
                setattr(os, _name, _value)
