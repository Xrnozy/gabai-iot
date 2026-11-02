"""TSDF fusion of the pipeline's depth (Open3D ScalableTSDFVolume), drawn as a mesh next to the voxel map.

Each frame's depth is put into a signed-distance volume with the camera pose; many looks at a surface average out the random depth error, so a wall comes out as one thin
surface. It sits beside the voxel map (gabai/grid.py), it does not replace it. It needs a good pose: with a drifting pose the surface is thick or doubled, whatever the fusion.
"""
import numpy as np

DEPTH_STD_PX = 0.3                # the stereo model's disparity noise; the same constant the voxel map uses for max_depth_std (z^2 * 0.3 / (f * B))


def depth_limit(max_range, max_depth_std, f, B):
    """Farthest depth (m) whose depth std stays under max_depth_std (0 = no std limit)."""
    z = float(max_range)
    if max_depth_std > 0:
        z = min(z, float(np.sqrt(max_depth_std * f * B / DEPTH_STD_PX)))
    return z


class TsdfMap:
    def __init__(self, cal, size, voxel=0.02, trunc=0.08, max_range=4.0, max_depth_std=0.0):
        import open3d as o3d
        self.o3d, self.n = o3d, 0
        self.voxel, self.trunc = voxel, trunc
        w, h = size
        self.f, self.B = float(cal["f"]), float(cal["B"])
        self.K = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), self.f, self.f, float(cal["cx"]), float(cal["cy"]))
        self.z_max = depth_limit(max_range, max_depth_std, self.f, self.B)
        self.reset()

    def reset(self):
        self.vol = self.o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=self.voxel, sdf_trunc=self.trunc,
                                                                      color_type=self.o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.n = 0

    def add(self, left_bgr, disp, trusted, R_wc, t_wc):
        """Fuse one frame. Only trusted pixels with a depth under the limit count; the rest are left empty (not 'free')."""
        o3d = self.o3d
        ok = trusted & np.isfinite(disp) & (disp > 0.5)
        z = np.zeros(disp.shape, np.float32)
        z[ok] = self.f * self.B / disp[ok]
        z[z > self.z_max] = 0.0
        rgb = np.ascontiguousarray(left_bgr[..., ::-1]) if left_bgr.ndim == 3 else np.ascontiguousarray(np.repeat(left_bgr[..., None], 3, axis=2))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(z), depth_scale=1.0, depth_trunc=self.z_max, convert_rgb_to_intensity=False)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.asarray(R_wc, float), np.asarray(t_wc, float)
        self.vol.integrate(rgbd, self.K, np.linalg.inv(T))                  # Open3D wants world -> camera
        self.n += 1

    def mesh(self):
        """(vertices, triangles, colors 0-255) or None while the volume is empty."""
        m = self.vol.extract_triangle_mesh()
        if len(m.vertices) == 0:
            return None
        return np.asarray(m.vertices, np.float32), np.asarray(m.triangles, np.uint32), (np.asarray(m.vertex_colors) * 255).astype(np.uint8)
