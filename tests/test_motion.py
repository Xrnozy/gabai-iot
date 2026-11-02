import math
import struct

from gabai.motion import turn_flag, sync_error_px
from gabai.pi.imu import decode_gyro, decode_accel, peak_speed


def test_decode_gyro_scales_raw_counts_to_degrees_per_second():
    raw = struct.pack("<3h", 16384, -16384, 0)         # half of full scale
    x, y, z = decode_gyro(raw, range_dps=500)
    assert math.isclose(x, 250.0, rel_tol=1e-6) and math.isclose(y, -250.0, rel_tol=1e-6) and z == 0.0


def test_decode_accel_one_g_at_default_range():
    raw = struct.pack("<3h", 0, 0, 16384)               # +-2 g range: 16384 counts = 1 g
    assert decode_accel(raw, range_g=2) == (0.0, 0.0, 1.0)


def test_peak_speed_uses_vector_magnitude_over_recent_window():
    samples = [(0.0, (0, 0, 0)), (0.1, (30, 40, 0)), (0.2, (0, 0, 10)), (1.0, (0, 0, 0))]
    assert math.isclose(peak_speed(samples, now=0.3, window=0.3), 50.0)   # |(30,40,0)|
    assert peak_speed(samples, now=1.2, window=0.3) == 0.0                # old samples dropped


def test_turn_flag_levels():
    assert turn_flag(5.0) == "steady"
    assert turn_flag(35.0) == "turning"
    assert turn_flag(90.0) == "fast"


def test_sync_error_is_focal_times_angle_moved_during_the_offset():
    # 100 deg/s for 17 ms with f = 881 px: 881 * 0.01745 * 1.745... = about 26 px
    e = sync_error_px(881.0, 100.0, 0.017)
    assert 25.0 < e < 27.0
    assert sync_error_px(881.0, 0.0, 0.017) == 0.0


def test_new_samples_since_picks_only_unsent_samples():
    from gabai.pi.imu import new_samples_since
    buf = [(1.0, 0, 0, 0, 0, 0, 1), (2.0, 0, 0, 0, 0, 0, 1), (3.0, 0, 0, 0, 0, 0, 1)]
    assert [s[0] for s in new_samples_since(buf, 1.5)] == [2.0, 3.0]
    assert new_samples_since(buf, 3.0) == []
    assert len(new_samples_since(buf * 100, 0.0, limit=50)) == 50      # never flood one message
