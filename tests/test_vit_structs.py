import ctypes as C

import pytest

from gabai.slam.vit import VitConfig, VitImgSample, VitImuSample, VitMask, VitPoseData


def test_struct_sizes_match_the_c_layout_of_vit_interface_h():
    assert C.sizeof(VitImuSample) == 32               # int64 + 6 floats
    assert C.sizeof(VitMask) == 16
    assert C.sizeof(VitPoseData) == 48                # int64 + 10 floats, padded to 8
    assert C.sizeof(VitConfig) == 24                  # pointer, two uint32, bool, padded
    assert C.sizeof(VitImgSample) == 56               # uint32 + pad, int64, pointer, 4 x uint32, enum, uint32, pointer


def test_field_offsets_that_the_dll_reads():
    assert VitImgSample.timestamp.offset == 8 and VitImgSample.data.offset == 16
    assert VitImgSample.width.offset == 24 and VitImgSample.format.offset == 40 and VitImgSample.masks.offset == 48
    assert VitPoseData.px.offset == 8 and VitPoseData.vx.offset == 36


@pytest.mark.skipif(not __import__("os").path.exists(r"D:\slam\basalt\build\basalt.dll"), reason="Basalt is not built here")
def test_the_basalt_dll_exports_the_tracker_functions():
    from gabai.slam.vit import load_dll
    dll = load_dll()
    for name in ("vit_api_get_version", "vit_tracker_create", "vit_tracker_start", "vit_tracker_push_imu_sample",
                 "vit_tracker_push_img_sample", "vit_tracker_pop_pose", "vit_pose_get_data", "vit_pose_destroy"):
        assert hasattr(dll, name), name
    v = [C.c_uint32() for _ in range(3)]
    assert dll.vit_api_get_version(*[C.byref(x) for x in v]) == 0 and v[0].value == 2


def test_the_basalt_config_file_goes_somewhere_without_spaces_even_for_a_folder_with_spaces():
    from gabai.tools.live_basalt import safe_toml_path
    p = safe_toml_path(r"C:\Users\MY PC\Documents\Hackathon testing\Gabai IOT final\data\slam\live_20261026_180602")
    assert " " not in p and p.endswith("live_20261026_180602.toml")


def test_stage_durations_turn_basalt_timestamps_into_milliseconds():
    from gabai.slam.vit import stage_durations_ms
    titles = ["frames_pushed", "frontend_frames_received", "frontend_keypoints_pushed", "backend_keypoints_received", "backend_state_pushed",
              "vit_consumer_pose_received"]
    ns = [0, 2_000_000, 7_000_000, 7_500_000, 16_000_000, 16_400_000]
    d = stage_durations_ms(titles, ns)
    assert d["queue_wait_ms"] == 2.0 and d["frontend_ms"] == 5.0 and d["backend_ms"] == 8.5
    assert d["basalt_total_ms"] == 14.0 and d["to_consumer_ms"] == 0.4
    assert stage_durations_ms(["frames_pushed"], [0])["backend_ms"] is None             # missing stamps are None, not 0
