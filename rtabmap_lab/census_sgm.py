"""Classical stereo matcher: census transform + semi-global matching, on the GPU (PyTorch). A second classical matcher next to OpenCV's StereoSGBM.

Why it can beat StereoSGBM here: the two cameras are different sensors with different exposure/gain, and census compares the ORDER of intensities in a window, not the intensities,
so a gain or brightness difference does not move the cost. SGM then makes the disparity smooth where the picture is flat and lets it jump where the picture has an edge (the smoothness penalty P2 is cut at intensity edges).
Steps: census 9x7 (62 bits) -> Hamming cost volume -> 8-path SGM aggregation -> winner-take-all -> uniqueness test -> left-right consistency -> parabola sub-pixel -> speckle removal.

    m = CensusSGM(max_disp=96); disp = m.disparity(left_gray_u8, right_gray_u8)     # float32 pixels, -1 where not trusted
"""
import cv2
import numpy as np
import torch

_POP = None


def _popcount_table(dev):
    global _POP
    if _POP is None or _POP.device != dev:
        _POP = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8, device=dev)
    return _POP


class CensusSGM:
    def __init__(self, max_disp=96, tol=15.0, p1=40.0, p2=500.0, p2_edge=150.0, edge_thr=8.0, uniq=0.0, lr_tol=1.0, speckle=60, paths=8, device="cuda"):
        self.tol, self.nd, self.p1, self.p2, self.p2e, self.thr = tol, max_disp, p1, p2, p2_edge, edge_thr
        self.uniq, self.lr_tol, self.speckle, self.paths, self.dev = uniq, lr_tol, speckle, paths, torch.device(device)

    def _census(self, g):
        """g: [H,W] float. Ternary census, 9 wide x 7 high without the centre: two int64 per pixel, bit k of the first = neighbour k is brighter than the centre by more than `tol`,
        of the second = darker by more than `tol`. Inside the tolerance (flat wall, sensor noise) both bits stay 0, so noise in a textureless area does not flip the code."""
        h, w = g.shape
        p = torch.nn.functional.pad(g[None, None], (4, 4, 3, 3), mode="replicate")[0, 0]
        hi = torch.zeros((h, w), dtype=torch.int64, device=g.device)
        lo = torch.zeros_like(hi)
        k = 0
        for dy in range(7):
            for dx in range(9):
                if dy == 3 and dx == 4:
                    continue
                n = p[dy:dy + h, dx:dx + w]
                hi |= (n > g + self.tol).to(torch.int64) << k
                lo |= (n < g - self.tol).to(torch.int64) << k
                k += 1
        return hi, lo

    def _volume(self, cl, cr):
        h, w = cl[0].shape
        pop = _popcount_table(cl[0].device)
        vol = torch.full((h, w, self.nd), 62.0, device=cl[0].device)             # x < d has no partner: worst cost
        for d in range(self.nd):
            c = 0
            for a, b in zip(cl, cr):
                c = c + pop[(a[:, d:] ^ b[:, :w - d]).view(torch.uint8).view(h, w - d, 8).long()].sum(-1)
            vol[:, d:, d] = c.float()
        return vol

    def _path(self, cost, img, dx):
        """One SGM path downwards over axis 0; dx = -1/0/1 shifts the predecessor sideways for diagonals."""
        out = torch.empty_like(cost)
        out[0] = cost[0]
        n, nd = cost.shape[1], cost.shape[2]
        inf = torch.full((n, 1), 1e9, device=cost.device)
        z = torch.zeros((1, nd), device=cost.device)
        for y in range(1, cost.shape[0]):
            prev, ip = out[y - 1], img[y - 1]
            if dx == 1:                                                      # predecessor of pixel x is x - dx on the row before
                prev, ip = torch.cat([z, prev[:-1]]), torch.cat([ip[:1], ip[:-1]])
            elif dx == -1:
                prev, ip = torch.cat([prev[1:], z]), torch.cat([ip[1:], ip[-1:]])
            m = prev.min(-1, keepdim=True).values
            p2 = torch.where((img[y] - ip).abs().unsqueeze(-1) > self.thr, self.p2e, self.p2)
            up = torch.cat([prev[:, 1:], inf], 1) + self.p1
            dn = torch.cat([inf, prev[:, :-1]], 1) + self.p1
            out[y] = cost[y] + torch.minimum(torch.minimum(prev, up), torch.minimum(dn, m + p2)) - m
        return out

    def disparity(self, left, right):
        with torch.no_grad():
            gl = torch.as_tensor(left, device=self.dev).float()
            gr = torch.as_tensor(right, device=self.dev).float()
            vol = self._volume(self._census(gl), self._census(gr))
            # (transpose, flip, dx): down, up, right, left, and the four diagonals
            runs = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, -1), (0, 1, 1), (0, 0, -1)] if self.paths == 8 else [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
            if self.paths == 8:
                runs = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 0, -1), (0, 1, 1), (0, 1, -1)]
            agg = torch.zeros_like(vol)
            for tr, fl, dx in runs:
                v, g = (vol.transpose(0, 1), gl.t()) if tr else (vol, gl)
                if fl:
                    v, g = v.flip(0), g.flip(0)
                o = self._path(v, g, dx)
                o = o.flip(0) if fl else o
                agg += o.transpose(0, 1) if tr else o
            return self._pick(agg)

    def _pick(self, s):
        h, w, nd = s.shape
        s1, d0 = s.min(-1)
        ar = torch.arange(nd, device=s.device)
        s2 = s.masked_fill((ar[None, None] - d0[..., None]).abs() <= 1, float("inf")).min(-1).values
        ok = (s2 - s1) > self.uniq * s1.clamp(min=1.0)                       # unique minimum
        # right-view disparity from the same volume: SR[y, x, d] = S[y, x + d, d]
        sr = torch.full_like(s, 1e12)
        for d in range(nd):
            sr[:, :w - d, d] = s[:, d:, d]
        dr = sr.argmin(-1)
        xs = (torch.arange(w, device=s.device)[None] - d0).clamp(min=0)
        ok &= (d0 - dr.gather(1, xs)).abs() <= self.lr_tol
        # parabola through the minimum and its neighbours
        i = d0.clamp(1, nd - 2)
        c0, cm, cp = s.gather(2, i[..., None])[..., 0], s.gather(2, (i - 1)[..., None])[..., 0], s.gather(2, (i + 1)[..., None])[..., 0]
        den = (cm - 2 * c0 + cp)
        sub = torch.where(den > 1e-6, 0.5 * (cm - cp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        d = (i + sub).float()
        edge = (d0 < 1) | (d0 > nd - 2)
        d = torch.where(ok & ~edge, d, torch.full_like(d, -1.0))
        out = d.cpu().numpy().astype(np.float32)
        if self.speckle:
            q = np.where(out > 0, out * 16, -16).astype(np.int16)
            cv2.filterSpeckles(q, -16, self.speckle, 16)
            out = np.where(q > 0, out, -1).astype(np.float32)
        return out
