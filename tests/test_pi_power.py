from gabai.pi.cams import wait_until_due
from gabai.pi.sender import SlowCache, low_power_settings


def test_the_camera_loop_sleeps_until_the_next_pair_is_due_instead_of_polling_every_millisecond():
    assert wait_until_due(now=10.000, last_out=10.000, min_gap=1 / 15) == 0.05     # just sent one: sleeps (the 50 ms cap, then re-checks)
    assert wait_until_due(now=10.060, last_out=10.000, min_gap=1 / 15) < 0.01      # nearly due: wake up soon
    assert wait_until_due(now=10.100, last_out=10.000, min_gap=1 / 15) == 0.0      # due: look now


def test_the_wait_is_capped_so_a_stop_request_is_still_noticed():
    assert wait_until_due(now=0.0, last_out=0.0, min_gap=5.0) <= 0.05


def test_slow_health_values_are_read_once_per_interval_not_every_second():
    calls = []
    c = SlowCache(lambda: calls.append(1) or {"temp_c": 50.0}, max_age=5.0)
    for now in (0.0, 1.0, 2.0, 4.9):
        assert c.get(now) == {"temp_c": 50.0}
    assert len(calls) == 1
    c.get(5.1)
    assert len(calls) == 2


def test_low_power_lowers_polling_but_not_below_what_the_sync_needs():
    n = low_power_settings(False, capture_fps=30)
    lp = low_power_settings(True, capture_fps=30)
    assert n == dict(capture_fps=30, imu_hz=200, mag_hz=50, health_every=1.0)
    assert lp["imu_hz"] == 100 and lp["mag_hz"] == 25 and lp["health_every"] == 5.0
    assert lp["capture_fps"] == 20                                               # worst-case pairing gap grows from about 17 ms to 25 ms (half a frame period); the sync error is stated in the guide
    assert low_power_settings(True, capture_fps=15)["capture_fps"] == 15          # never raised above what the user asked for
