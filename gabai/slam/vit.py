"""Python access to Basalt's live tracker (the VIT C interface in D:\\slam\\basalt\\build\\basalt.dll).

    t = BasaltTracker("D:/path/basalt.toml")
    t.push_imu(t_ns, accel_m_s2, gyro_rad_s)       # in time order
    t.push_frame(t_ns, left_gray, right_gray)      # same stamp for both cameras, strictly increasing
    pose = t.pop_pose()                            # None until Basalt has one

The structs mirror thirdparty/vit/vit_interface.h (version 2.0.1). Basalt aborts the whole process if frames arrive out of
camera order or with different stamps, so push_frame always sends camera 0 then camera 1 with one stamp.
"""
import ctypes as C
import os

import numpy as np

BASALT_DLL = r"D:\slam\basalt\build\basalt.dll"
DLL_DIRS = (r"D:\slam\vcpkg\installed\x64-windows\bin", r"D:\slam\basalt\build")
VIT_SUCCESS = 0
FORMAT_L8 = 1


class VitConfig(C.Structure):
    _fields_ = [("file", C.c_char_p), ("cam_count", C.c_uint32), ("imu_count", C.c_uint32), ("show_ui", C.c_bool)]


class VitImuSample(C.Structure):
    _fields_ = [("timestamp", C.c_int64), ("ax", C.c_float), ("ay", C.c_float), ("az", C.c_float),
                ("wx", C.c_float), ("wy", C.c_float), ("wz", C.c_float)]


class VitMask(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("w", C.c_float), ("h", C.c_float)]


class VitImgSample(C.Structure):
    _fields_ = [("cam_index", C.c_uint32), ("timestamp", C.c_int64), ("data", C.POINTER(C.c_uint8)),
                ("width", C.c_uint32), ("height", C.c_uint32), ("stride", C.c_uint32), ("size", C.c_uint32),
                ("format", C.c_int), ("mask_count", C.c_uint32), ("masks", C.POINTER(VitMask))]


class VitPoseData(C.Structure):
    _fields_ = [("timestamp", C.c_int64), ("px", C.c_float), ("py", C.c_float), ("pz", C.c_float),
                ("ox", C.c_float), ("oy", C.c_float), ("oz", C.c_float), ("ow", C.c_float),
                ("vx", C.c_float), ("vy", C.c_float), ("vz", C.c_float)]


class VitTimingTitles(C.Structure):
    _fields_ = [("count", C.c_uint32), ("titles", C.POINTER(C.c_char_p))]


class VitPoseTiming(C.Structure):
    _fields_ = [("count", C.c_uint32), ("timestamps", C.POINTER(C.c_int64))]


EXT_POSE_TIMING = 2


def stage_durations_ms(titles, stamps):
    """Basalt's per-frame stage timestamps (ns, steady clock) -> milliseconds spent in each part. A part whose two stamps are missing is None."""
    t = dict(zip(titles, stamps))

    def d(a, b):
        return round((t[b] - t[a]) / 1e6, 3) if a in t and b in t else None
    return {"queue_wait_ms": d("frames_pushed", "frontend_frames_received"), "frontend_ms": d("frontend_frames_received", "frontend_keypoints_pushed"),
            "backend_ms": d("backend_keypoints_received", "backend_state_pushed"), "basalt_total_ms": d("frontend_frames_received", "backend_state_pushed"),
            "to_consumer_ms": d("backend_state_pushed", "vit_consumer_pose_received")}


def load_dll(path=BASALT_DLL):
    for d in DLL_DIRS:
        if os.path.isdir(d):
            os.add_dll_directory(d)
    return C.CDLL(path)


class BasaltTracker:
    def __init__(self, config_file, show_ui=False, dll=None):
        self.dll = dll or load_dll()
        major, minor, patch = (C.c_uint32() for _ in range(3))
        self._check(self.dll.vit_api_get_version(C.byref(major), C.byref(minor), C.byref(patch)), "vit_api_get_version")
        if major.value != 2:
            raise RuntimeError(f"Basalt speaks VIT {major.value}.{minor.value}, this wrapper is written for 2.x")
        cfg = VitConfig(config_file.encode(), 2, 1, show_ui)
        self.h = C.c_void_p()
        self._check(self.dll.vit_tracker_create(C.byref(cfg), C.byref(self.h)), "vit_tracker_create")
        self._check(self.dll.vit_tracker_start(self.h), "vit_tracker_start")
        self.last_frame_ns = -1
        self.last_imu_ns = -1
        self.titles = None
        self.timing_enabled = False

    @staticmethod
    def _check(r, what):
        if r != VIT_SUCCESS:
            raise RuntimeError(f"{what} failed with VIT result {r}")

    def enable_timing(self):
        """Ask Basalt for its per-stage timestamps with every pose (pose["timing"]). Returns False if this build does not offer it."""
        if self.dll.vit_tracker_enable_extension(self.h, EXT_POSE_TIMING, True) != VIT_SUCCESS:
            return False
        ti = VitTimingTitles()
        if self.dll.vit_tracker_get_timing_titles(self.h, C.byref(ti)) != VIT_SUCCESS:
            return False
        self.titles = [ti.titles[i].decode() for i in range(ti.count)]
        self.timing_enabled = True
        return True

    def push_imu(self, t_ns, accel, gyro):
        """accel in m/s^2, gyro in rad/s. Returns False (nothing pushed) if the stamp does not advance."""
        if t_ns <= self.last_imu_ns:
            return False
        s = VitImuSample(int(t_ns), *map(float, accel), *map(float, gyro))
        self._check(self.dll.vit_tracker_push_imu_sample(self.h, C.byref(s)), "push_imu_sample")
        self.last_imu_ns = t_ns
        return True

    def push_frame(self, t_ns, left, right):
        """left, right: 2-D uint8 arrays of the same size. Returns False if the stamp does not advance."""
        if t_ns <= self.last_frame_ns:
            return False
        for cam, img in enumerate((left, right)):
            img = np.ascontiguousarray(img, dtype=np.uint8)
            if img.ndim != 2:
                raise ValueError("frames must be single-channel")
            h, w = img.shape
            s = VitImgSample(cam, int(t_ns), img.ctypes.data_as(C.POINTER(C.c_uint8)), w, h, w, w * h, FORMAT_L8, 0, None)
            self._check(self.dll.vit_tracker_push_img_sample(self.h, C.byref(s)), "push_img_sample")
        self.last_frame_ns = t_ns
        return True

    def pop_pose(self):
        """Next pose as a dict (t_ns, p, q xyzw, v), or None when none is ready."""
        pose = C.c_void_p()
        self._check(self.dll.vit_tracker_pop_pose(self.h, C.byref(pose)), "pop_pose")
        if not pose.value:
            return None
        d = VitPoseData()
        self._check(self.dll.vit_pose_get_data(pose, C.byref(d)), "pose_get_data")
        timing = None
        if self.timing_enabled:
            tm = VitPoseTiming()
            if self.dll.vit_pose_get_timing(pose, C.byref(tm)) == VIT_SUCCESS and tm.count == len(self.titles):
                timing = stage_durations_ms(self.titles, [tm.timestamps[i] for i in range(tm.count)])
        self.dll.vit_pose_destroy(pose)
        return {"t_ns": d.timestamp, "p": (d.px, d.py, d.pz), "q": (d.ox, d.oy, d.oz, d.ow), "v": (d.vx, d.vy, d.vz), "timing": timing}

    def close(self):
        if self.h:
            self.dll.vit_tracker_stop(self.h)
            self.dll.vit_tracker_destroy(self.h)
            self.h = C.c_void_p()
