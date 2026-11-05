"""Per-pixel depth confidence in [0, 1] from one stereo pair's two disparity maps.

conf = left-right agreement x texture x edge x range. Each factor answers one way a disparity can be wrong:
  agreement  the right-view disparity, read where the left pixel lands, says the same thing (occlusions and mismatches disagree)
  texture    a blank patch gives the network nothing to measure; it still outputs a plausible number, so it is trusted less
  edge       a pixel next to a disparity jump bigger than the noise explains is a flying pixel
  range      sigma_z = z^2 * sigma_d / (f B) against the noise the map can use; zero past max_range
All torch, runs on whatever device the inputs are on.
"""
import torch
import torch.nn.functional as F

SIGMA_D = 0.5       # px, measured disparity noise (docs/rtabmap-noise.md)
SIGMA_Z_MAX = 0.14  # m, noise at which the range factor has fallen to 0.6 (was 0.10: it cut the far half of the cloud although the depth there is right)
LR_SIGMA = 0.8      # px, left-right disagreement that still counts as agreement (was 0.5 = the disparity noise: half of the correct pixels at edges and fine detail failed it)
EDGE_JUMP = 3.0     # px, disparity spread over 3x3 that is a real edge before the pixel is penalised (was 1.5: it erased thin objects and every object outline)
TEXTURE_FLOOR = 0.5  # weight of a blank patch (was 0.3)


def _smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def lr_agreement(dl, dr, sigma=LR_SIGMA):
    """dl, dr: (H, W) disparity of the left and of the right view. Right disparity is sampled at x - dl."""
    h, w = dl.shape
    xs = torch.arange(w, device=dl.device, dtype=dl.dtype)[None, :] - dl
    gx = xs / (w - 1) * 2 - 1
    gy = (torch.arange(h, device=dl.device, dtype=dl.dtype)[:, None] / (h - 1) * 2 - 1).expand(h, w)
    dr_at = F.grid_sample(dr[None, None], torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    inside = (xs >= 0) & (xs <= w - 1)
    return torch.where(inside, torch.exp(-0.5 * ((dl - dr_at) / sigma) ** 2), torch.zeros_like(dl))


def texture(gray, lo=2.0, hi=8.0):
    """gray: (H, W) float 0..255. Mean gradient magnitude over 7x7; TEXTURE_FLOOR so a blank-but-consistent wall is not thrown away."""
    g = gray[None, None]
    gx = g[..., :, 1:] - g[..., :, :-1]
    gy = g[..., 1:, :] - g[..., :-1, :]
    mag = F.pad(gx.abs(), (0, 1, 0, 0)) + F.pad(gy.abs(), (0, 0, 0, 1))
    return TEXTURE_FLOOR + (1 - TEXTURE_FLOOR) * _smoothstep(F.avg_pool2d(mag, 7, 1, 3)[0, 0], lo, hi)


def edge(d, jump=EDGE_JUMP):
    """Penalise pixels whose 3x3 neighbourhood spans more than `jump` px of disparity."""
    x = d[None, None]
    spread = (F.max_pool2d(x, 3, 1, 1) - (-F.max_pool2d(-x, 3, 1, 1)))[0, 0]
    return 1.0 / (1.0 + (spread / jump) ** 2)


def confidence(dl, dr, gray, fb, max_range=2.5, sigma_d=SIGMA_D):
    """Returns (depth m, confidence, sigma_z m). fb = focal px x baseline m. Depth is 0 where disparity is not positive or past max_range."""
    valid = dl > 0.5
    z = torch.where(valid, fb / dl.clamp(min=0.5), torch.zeros_like(dl))
    sigma_z = z * z * sigma_d / fb
    rng = torch.exp(-0.5 * (sigma_z / SIGMA_Z_MAX) ** 2)
    c = lr_agreement(dl, dr) * texture(gray) * edge(dl) * rng
    c = torch.where(valid & (z <= max_range), c, torch.zeros_like(c))
    z = torch.where(c > 0, z, torch.zeros_like(z))
    return z, c, sigma_z
