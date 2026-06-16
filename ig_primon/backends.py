"""Backend abstraction over the real heterogeneous hardware.

A Backend is one (device, array-library) pair: the CPU via numpy, or each CUDA
device via cupy. The precision-matrix (ig_primon.precision) runs the same operation
across {backend} x {dtype} and certifies each cell against a Tier-C fp64 reference --
the hw:sw abstraction made measurable.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Backend:
    id: str          # "cpu", "cuda0", "cuda1"
    name: str        # "CPU (numpy)", "NVIDIA GeForce RTX 5070"
    kind: str        # "cpu" | "cuda"
    device: int      # cuda device index, or -1 for cpu
    cc: str          # "sm_120" for cuda, "" for cpu

    # dtypes worth measuring on this backend (fp16 on the CPU is pointlessly slow)
    @property
    def dtypes(self):
        return ["fp64", "fp32", "fp16"] if self.kind == "cuda" else ["fp64", "fp32"]


def discover():
    """Return the CPU backend plus one Backend per visible CUDA device."""
    backends = [Backend("cpu", "CPU (numpy)", "cpu", -1, "")]
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
            for d in range(cp.cuda.runtime.getDeviceCount()):
                name = cp.cuda.runtime.getDeviceProperties(d)["name"].decode()
                with cp.cuda.Device(d):
                    cc = cp.cuda.Device(d).compute_capability
                backends.append(Backend(f"cuda{d}", name, "cuda", d, f"sm_{cc}"))
    except Exception:
        pass
    return backends


def get_xp(backend: Backend):
    """The array library for a backend: cupy (cuda) or numpy (cpu)."""
    if backend.kind == "cuda":
        import cupy as cp
        return cp
    return np


def on_device(backend: Backend):
    """Context manager selecting the CUDA device (no-op for the CPU)."""
    if backend.kind == "cuda":
        import cupy as cp
        return cp.cuda.Device(backend.device)
    return contextlib.nullcontext()


def sync(backend: Backend):
    """Block until queued work on the backend has finished (no-op for the CPU)."""
    if backend.kind == "cuda":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


# numpy dtypes by short name
NP_DTYPE = {"fp64": np.float64, "fp32": np.float32, "fp16": np.float16}
