"""Configures and builds rtabmap_lab/bridge with CMake + Ninja inside the Visual Studio Build Tools environment.

  py rtabmap_lab/tools/build_bridge.py          -> rtabmap_lab/bin/rtabmap_bridge.exe
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
VCVARS = os.environ.get("VCVARS", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def main():
    if not os.path.exists(VCVARS):
        sys.exit(f"vcvars64.bat not found: {VCVARS}")
    build = os.path.join(LAB, "bridge", "build")
    src = os.path.join(LAB, "bridge")
    cmd = f'cmd /c ""{VCVARS}" >nul && cmake -S "{src}" -B "{build}" -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build "{build}""'
    r = subprocess.run(cmd, shell=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
