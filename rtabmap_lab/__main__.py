"""py -m rtabmap_lab <command>

  setup                 download the OpenCV 4.7.0 / Eigen headers, make import libraries from the installed RTAB-Map, build rtabmap_bridge.exe (once)
  live [options]        RTAB-Map on the Pi's stereo stream, dashboard in Rerun
  replay --folder DIR   the same on a recording made by gabai.tools.record_pairs
  export DB             write the map in DB as a point cloud (.ply) and the poses with RTAB-Map's own exporter
  stop                  kill a bridge that was left running
"""
import os
import subprocess
import sys

from rtabmap_lab.bridge_client import RTABMAP_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "setup":
        for tool in ("fetch_deps", "make_importlibs", "build_bridge"):
            r = subprocess.run([sys.executable, os.path.join(HERE, "tools", tool + ".py")])
            if r.returncode:
                sys.exit(r.returncode)
    elif cmd in ("live", "replay"):
        from rtabmap_lab.live import main as live
        live(["--source", cmd] + rest)
    elif cmd == "export":
        if not rest:
            sys.exit("py -m rtabmap_lab export path/to/map.db [more rtabmap-export options]")
        exe = os.path.join(RTABMAP_DIR, "bin", "rtabmap-export.exe")
        env = dict(os.environ, PATH=os.path.join(RTABMAP_DIR, "bin") + os.pathsep + os.environ.get("PATH", ""))
        sys.exit(subprocess.run([exe, "--cloud", "--poses", "--output_dir", os.path.dirname(os.path.abspath(rest[0]))] + rest[1:] + [rest[0]], env=env).returncode)
    elif cmd == "stop":
        r = subprocess.run(["taskkill", "/F", "/IM", "rtabmap_bridge.exe"], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
