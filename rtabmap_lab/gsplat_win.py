"""Makes gsplat build and import on this Windows PC (torch built for CUDA 12.8, toolkit 13.1, MSVC Build Tools 2022). Import this module BEFORE gsplat.

What it does, and why each step is needed (found by building gsplat 1.5.3 here):
  - vcvars64.bat is run once in this process's environment when cl.exe is not on PATH (gsplat compiles itself with ninja on first use, 5-10 minutes the first time, cached afterwards).
  - WIN32_LEAN_AND_MEAN: without it windows.h pulls in rpcndr.h, whose  #define small char  breaks torch's header c10/cuda/CUDACachingAllocator.h ("bool small").
  - PyTorch's own check refuses a toolkit whose major version differs from the one torch was built with (12.8 against 13.1); the toolkit's compiler builds the extension fine, so the check is skipped.
  - gsplat passes GCC flags (-O3, -Wno-attributes) to cl.exe, which rejects them: they are replaced by /O2 in the installed gsplat (an idempotent one-line edit of gsplat/cuda/_backend.py).
  - TORCH_CUDA_ARCH_LIST=12.0 (RTX 50 series) so only one architecture is compiled.
"""
import os
import re
import subprocess

VCVARS = os.environ.get("VCVARS", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
CUDA_HOME = os.environ.get("CUDA_HOME") or r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"


def _vcvars():
    import shutil
    if shutil.which("cl") or not os.path.exists(VCVARS):
        return
    out = subprocess.run(f'cmd /c ""{VCVARS}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k and v and k.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VSINSTALLDIR", "WINDOWSSDKDIR", "UCRTVERSION", "WINDOWSSDKVERSION"):
            os.environ[k.upper() if k.upper() == "PATH" else k] = v


def _patch_gsplat():
    import importlib.util
    spec = importlib.util.find_spec("gsplat")
    path = os.path.join(os.path.dirname(spec.origin), "cuda", "_backend.py")
    s = open(path, encoding="utf-8").read()
    old = 'extra_cflags = [opt_level, "-Wno-attributes"]'
    if old in s:
        open(path, "w", encoding="utf-8").write(s.replace(old, 'extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]'))


def setup():
    if os.name != "nt":
        return
    os.environ.setdefault("CUDA_HOME", CUDA_HOME)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    for var, flags in (("NVCC_APPEND_FLAGS", "-DWIN32_LEAN_AND_MEAN -DNOMINMAX"), ("CL", "/DWIN32_LEAN_AND_MEAN /DNOMINMAX")):
        if "WIN32_LEAN_AND_MEAN" not in os.environ.get(var, ""):
            os.environ[var] = (os.environ.get(var, "") + " " + flags).strip()
    _vcvars()
    from torch.utils import cpp_extension
    cpp_extension._check_cuda_version = lambda *a, **k: None
    _patch_gsplat()


setup()
