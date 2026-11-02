"""Makes MSVC import libraries (.lib) for the OpenCV DLLs that ship inside the installed RTAB-Map (it ships the DLLs but no .lib), so the bridge links against exactly
the OpenCV that rtabmap_core.dll was built with.

  py rtabmap_lab/tools/make_importlibs.py            -> rtabmap_lab/deps/implibs/opencv_*470.lib
"""
import glob
import os
import re
import subprocess
import sys

RTABMAP = os.environ.get("RTABMAP_DIR", r"C:\Program Files\RTABMap")
VCVARS = os.environ.get("VCVARS", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "deps", "implibs")
WANT = ("core", "imgproc", "imgcodecs", "calib3d", "features2d", "flann", "video", "videoio", "highgui")


def run(cmd):
    r = subprocess.run(f'cmd /c ""{VCVARS}" >nul && {cmd}"', capture_output=True, text=True, shell=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    if not os.path.exists(VCVARS):
        sys.exit(f"vcvars64.bat not found: {VCVARS} (set VCVARS to the Visual Studio Build Tools one)")
    os.makedirs(OUT, exist_ok=True)
    for dll in sorted(glob.glob(os.path.join(RTABMAP, "bin", "opencv_*470.dll"))):
        name = os.path.splitext(os.path.basename(dll))[0]
        if not any(name.startswith(f"opencv_{w}4") for w in WANT):
            continue
        text = run(f'dumpbin /exports "{dll}"')
        syms = re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)", text, re.M)
        if not syms:
            sys.exit(f"no exports found in {dll}")
        deff = os.path.join(OUT, name + ".def")
        with open(deff, "w") as f:
            f.write(f"LIBRARY {name}\nEXPORTS\n" + "\n".join(syms) + "\n")
        run(f'lib /nologo /def:"{deff}" /out:"{os.path.join(OUT, name + ".lib")}" /machine:x64')
        print(f"{name}.lib  ({len(syms)} exports)")


if __name__ == "__main__":
    main()
