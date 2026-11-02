"""Prepares rtabmap_lab/deps: the OpenCV 4.7.0 and Eigen 3.4.0 HEADERS the bridge compiles against, and a few empty stand-ins for PCL headers.

  py rtabmap_lab/tools/fetch_deps.py        downloads two zips (OpenCV 4.7.0 about 90 MB, Eigen 3.4.0 about 4 MB) into rtabmap_lab/deps/dl, extracts only headers

Why these: the installed RTAB-Map ships its DLLs and import libraries but no headers of OpenCV or Eigen. 4.7.0 is the OpenCV that rtabmap_core.dll was built with
(opencv_*470.dll in its bin folder). PCL: the RTAB-Map headers the bridge includes mention PCL but declare no PCL type, so empty stand-ins are enough to compile; nothing
of PCL is linked or called by the bridge.
"""
import os
import re
import shutil
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.abspath(os.path.join(HERE, "..", "deps"))
SOURCES = {"opencv-4.7.0.zip": "https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip",
           "eigen-3.4.0.zip": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"}
MODS = "calib3d core dnn features2d flann highgui imgcodecs imgproc objdetect photo stitching video videoio".split()
PCL_CONFIG = "#pragma once\n#define PCL_MAJOR_VERSION 1\n#define PCL_MINOR_VERSION 14\n#define PCL_REVISION_VERSION 1\n#define PCL_VERSION_COMPARE(OP, MAJ, MIN, PATCH) ((1 OP MAJ) || (1 == MAJ && (14 OP MIN || (14 == MIN && 1 OP PATCH))))\n"


def download():
    os.makedirs(os.path.join(DEPS, "dl"), exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(DEPS, "dl", name)
        if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            print("downloading", url)
            urllib.request.urlretrieve(url, path)


def extract():
    out_inc = os.path.join(DEPS, "opencv", "include")
    with zipfile.ZipFile(os.path.join(DEPS, "dl", "opencv-4.7.0.zip")) as z:
        for name in z.namelist():
            m = re.match(r"opencv-4\.7\.0/(?:include/|modules/([a-z0-9_]+)/include/)(.+)$", name)
            if not m or name.endswith("/") or (m.group(1) and m.group(1) not in MODS):
                continue
            dst = os.path.join(out_inc, m.group(2))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(name) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
    with open(os.path.join(out_inc, "opencv2", "opencv_modules.hpp"), "w") as f:
        f.write("#pragma once\n" + "".join(f"#define HAVE_OPENCV_{m.upper()}\n" for m in MODS))
    with zipfile.ZipFile(os.path.join(DEPS, "dl", "eigen-3.4.0.zip")) as z:
        for name in z.namelist():
            m = re.match(r"eigen-3\.4\.0/(Eigen/.+)$", name)
            if not m or name.endswith("/"):
                continue
            dst = os.path.join(DEPS, "eigen", m.group(1))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(name) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
    stub = "#pragma once\n// empty stand-in: the RTAB-Map headers used by the bridge include this but declare no PCL type\n"
    for p, text in {"pcl/point_types.h": stub, "pcl/point_cloud.h": stub, "pcl/pcl_base.h": stub, "pcl/pcl_macros.h": stub,
                    "pcl/types.h": "#pragma once\n#include <cstdint>\nnamespace pcl { using index_t = std::int32_t; }\n", "pcl/pcl_config.h": PCL_CONFIG}.items():
        d = os.path.join(DEPS, "stubs", p)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        open(d, "w").write(text)


if __name__ == "__main__":
    download()
    extract()
    print("deps ready in", DEPS, "- next: make_importlibs.py, build_bridge.py")
