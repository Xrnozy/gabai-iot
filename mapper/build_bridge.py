"""Builds mapper/bridge into mapper/bin/mapper_bridge.exe (same toolchain and deps as rtabmap_lab/tools/build_bridge.py)."""
import os
import subprocess
import sys

from rtabmap_lab.tools.build_bridge import VCVARS

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge")

if __name__ == "__main__":
    build = os.path.join(SRC, "build")
    sys.exit(subprocess.run(f'cmd /c ""{VCVARS}" >nul && cmake -S "{SRC}" -B "{build}" -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build "{build}""', shell=True).returncode)
