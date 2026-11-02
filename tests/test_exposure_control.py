import numpy as np

from gabai.pi import cams
from gabai.pi.cams import control_limits, next_settings

LIM = {"exposure": (1, 5000), "gain": (0, 100)}
CTRLS = """
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
                   auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
"""


def test_the_camera_limits_are_read_from_the_v4l2_listing():
    class R:
        stdout = CTRLS
    real = cams.subprocess.run
    cams.subprocess.run = lambda *a, **k: R()
    try:
        lim = control_limits("/dev/video0")
    finally:
        cams.subprocess.run = real
    assert lim["gain"] == (0, 100) and lim["exposure_time_absolute"] == (1, 5000) and lim["brightness"] == (-64, 64)


def _settle(scene, start=(40, 0), steps=60, **kw):
    """A fake camera: picture mean = scene * exposure * (1 + gain / 50), clipped. Runs the rule until it stops changing."""
    e, g = start
    starved = False
    for _ in range(steps):
        mean = min(255.0, scene * e * (1 + g / 50.0))
        e, g, starved = next_settings(mean, e, g, LIM, **kw)
    return e, g, min(255.0, scene * e * (1 + g / 50.0)), starved


def test_a_dim_room_gets_the_capped_exposure_and_the_gain_makes_up_the_rest():
    e, g, mean, starved = _settle(0.3)                                   # dim: needs more light than 8 ms alone gives
    assert e == 160 and 0 < g < 100 and abs(mean - 105) <= 25 and not starved   # 160 x 0.1 ms = 16 ms, never more


def test_a_room_that_is_too_dark_even_at_full_gain_is_reported_not_smeared():
    e, g, mean, starved = _settle(0.05)
    assert e == 160 and g == 100 and starved and mean < 93


def test_bright_light_lowers_the_gain_first_then_the_exposure():
    e, g, mean, _ = _settle(8.0, start=(160, 60))
    assert g == 0 and e < 160 and abs(mean - 105) <= 40
    e2, g2, _ = next_settings(250.0, 160, 60, LIM)
    assert g2 < 60 and e2 == 160                                          # gain moves before exposure


def test_the_exposure_never_passes_the_cap_whatever_the_picture_says():
    for mean in (0.0, 30.0, 90.0, 120.0, 250.0):
        e, g, _ = next_settings(mean, 79, 50, LIM, max_exposure_ms=8.0)
        assert e <= 80
    e, _, _ = next_settings(0.0, 40, 100, LIM, max_exposure_ms=3.0)
    assert e <= 30


def test_a_camera_without_a_gain_control_still_gets_a_capped_exposure():
    e, g, starved = next_settings(10.0, 160, 0, {"exposure": (1, 5000)})
    assert e == 160 and g == 0 and starved


def test_a_starved_pi_is_named_but_a_past_brown_out_is_not_reported_as_current():
    from gabai.pipeline.sources import power_warning
    w = power_warning(0x50005)
    assert "under_voltage_now" in w and "throttled_now" in w and "pi-power-guide" in w
    assert power_warning(0x50000) == "" and power_warning(0) == "" and power_warning(None) == ""
