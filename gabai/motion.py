"""Head-turn awareness. While the rig turns, the two cameras (exposing a few ms apart) see the scene shifted by
focal_length * angular_speed * time_offset pixels. That shift is read as extra disparity, so depth gets worse the
faster you turn. The thresholds are starting values to tune on real walking data."""
import math

TURNING_DPS = 25.0
FAST_DPS = 60.0
SYNC_TYPICAL_S = 0.004   # measured median left/right exposure offset (docs/spike-report.md)
SYNC_WORST_S = 0.022     # measured worst offset in 47 edges


def turn_flag(peak_dps):
    return "fast" if peak_dps >= FAST_DPS else "turning" if peak_dps >= TURNING_DPS else "steady"


def sync_error_px(focal_px, dps, offset_s):
    """Extra horizontal shift between the two images caused by turning during the offset between exposures."""
    return focal_px * math.radians(dps) * offset_s
