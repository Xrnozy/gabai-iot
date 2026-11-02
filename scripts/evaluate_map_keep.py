"""Does the map survive a stretch where the pose and the map disagree? Replays a recording and prints the map size over time, how often the map was dropped or the pose was fitted back
to it, and how much of the map's peak is left at the end.

py scripts/evaluate_map_keep.py data/recordings/rapid_20261029_235536 [--model h] [--set map.recover_every=0 map.forget_in_view=false ...]
"""
import argparse
import tempfile

import numpy as np

from evaluate_still import Log, parse
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ov = {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": a.folder, "source.max_frames": 0, "stereo.model": a.model, "stereo.mono": "", "viz.viewer": False,
          "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "tracking.gyro_prior": False}
    ov.update(parse(a.set))
    s = run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=Log, log_dir=tempfile.mkdtemp(), quiet=True)
    vox = np.array([x[1] for x in Log.last.sc.get("tracking/map_voxels", [])])
    ev = [e[1] for e in Log.last.events]
    fitted = sum("fitted back to it" in e for e in ev)
    dropped = sum("disagreed for" in e for e in ev)
    print(f"{' '.join(a.set) or 'defaults'}: frames {s['frames']} | map voxels peak {int(vox.max())}, end {int(vox[-1])} ({vox[-1] / vox.max() * 100:.0f}% of the peak), lowest after the peak {int(vox[int(vox.argmax()):].min())}"
          f" | map dropped {dropped}x, pose fitted back to the map {fitted}x | inconsistent frames {s['frames_skipped_inconsistent']} | align {s['map_alignment_median_cm']['median'] if s.get('map_alignment_median_cm') else None} cm")


if __name__ == "__main__":
    main()
