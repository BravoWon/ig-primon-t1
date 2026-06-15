"""Hardware scan + Tier assignment for the Precision-Certification Firewall.

The original hardware doctrine (``T1_hw_optimization_doctrine_v0_1.md``) targeted a
Snapdragon X Plus (Oryon ARM64 CPU + Hexagon NPU + Adreno GPU, no FP64 on the
accelerators). The real device is different and is detected here at runtime, so the
firewall always runs against the machine it is actually on.

Tier rule (unchanged from the doctrine):
  * Tier-C (Certify) = the CPU at arbitrary precision (mpmath). The SOLE [V] authority.
  * Tier-E (Explore) = the fastest available accelerator at reduced precision. Proposes
    candidates only; never awards [V].
On an NVIDIA box Tier-E is a CUDA GPU (FP32/TF32); with no GPU compute library it falls
back to CPU vectorised FP32. Either way the firewall's job is the same: a candidate is
trusted only after Tier-C reproduces it within budget.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field


@dataclass
class GPU:
    index: int
    name: str
    memory_mib: int
    driver: str
    compute_cap: str


@dataclass
class DeviceMap:
    cpu_name: str
    cpu_arch: str
    logical_cpus: int
    os_desc: str
    gpus: list = field(default_factory=list)
    cupy_available: bool = False
    cupy_device: str = ""
    cupy_cc: str = ""
    tier_c: str = ""
    tier_e: str = ""
    tier_e_backend: str = ""   # "cuda" | "cpu-fp32"
    notes: list = field(default_factory=list)


def _query_nvidia_smi():
    exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not (exe and os.path.exists(exe)):
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                gpus.append(GPU(int(parts[0]), parts[1], int(float(parts[2])),
                                parts[3], parts[4]))
            except ValueError:
                continue
    return gpus


def _probe_cupy():
    """Return (available, device_name, compute_capability). Imports cupy if installed."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # benign CUDA_PATH detection warning
            import cupy as cp
            dev = cp.cuda.Device(0)
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            cc = dev.compute_capability
            # confirm a kernel actually compiles+runs on this device (Blackwell needs NVRTC>=12.8)
            _ = float((cp.arange(16, dtype=cp.float32) ** 2).sum())
            return True, name, cc
    except Exception:
        return False, "", ""


def scan():
    """Detect the real device and assign the Tier-C / Tier-E split for this machine."""
    dm = DeviceMap(
        cpu_name=platform.processor() or platform.machine(),
        cpu_arch=platform.machine(),
        logical_cpus=os.cpu_count() or 0,
        os_desc=f"{platform.system()} {platform.release()} ({platform.version()})",
    )
    dm.gpus = _query_nvidia_smi()
    dm.cupy_available, dm.cupy_device, dm.cupy_cc = _probe_cupy()

    dm.tier_c = f"CPU {dm.cpu_name} - mpmath (sole [V] authority)"
    if dm.cupy_available:
        cc = dm.cupy_cc
        if isinstance(cc, tuple):
            cc_str, cc_num = f"sm_{cc[0]}{cc[1]}", cc[0] * 10 + cc[1]
        else:
            s = str(cc)                       # cupy returns e.g. "120" for 12.0
            cc_str, cc_num = f"sm_{s}", int(s)
        dm.tier_e = f"CUDA GPU {dm.cupy_device} ({cc_str}) - FP32 explorer"
        dm.tier_e_backend = "cuda"
        if cc_num >= 120:
            dm.notes.append("Blackwell-class GPU (sm_120): doctrine NO-FP64-ACCEL / NPU-FORMAT walls obsolete.")
    elif dm.gpus:
        dm.tier_e = "CPU vectorised FP32 (NVIDIA GPU present; install ig-primon-t1[gpu] to use it)"
        dm.tier_e_backend = "cpu-fp32"
        dm.notes.append("GPU detected but no CuPy: 'pip install cupy-cuda12x[ctk]' to light up Tier-E.")
    else:
        dm.tier_e = "CPU vectorised FP32 (no accelerator detected)"
        dm.tier_e_backend = "cpu-fp32"
    return dm


def format_scan(dm: DeviceMap):
    lines = []
    lines.append("=" * 84)
    lines.append("IG-PRIMON-T1 - device scan & Precision-Certification Firewall tier map")
    lines.append("=" * 84)
    lines.append(f"  CPU : {dm.cpu_name}  [{dm.cpu_arch}]  {dm.logical_cpus} logical")
    lines.append(f"  OS  : {dm.os_desc}")
    if dm.gpus:
        lines.append("  GPU :")
        for g in dm.gpus:
            lines.append(f"        [{g.index}] {g.name}  {g.memory_mib} MiB  "
                         f"cc {g.compute_cap}  driver {g.driver}")
    else:
        lines.append("  GPU : none detected")
    lines.append("")
    lines.append(f"  Tier-C (Certify, sole [V]) : {dm.tier_c}")
    lines.append(f"  Tier-E (Explore, [E-hw])   : {dm.tier_e}")
    lines.append(f"  Tier-E backend             : {dm.tier_e_backend}")
    for n in dm.notes:
        lines.append(f"  note: {n}")
    lines.append("=" * 84)
    return "\n".join(lines)
