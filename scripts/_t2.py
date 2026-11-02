import sys; sys.argv=['x']
from scripts.classical_bench import *
import rtabmap_lab.census_sgm as cs
pairs=list(gt_pairs()); l,r,m_,gt,_=pairs[0]
m=CensusSGM(ND,uniq=0.0,lr_tol=1e9,speckle=0)
d=m.disparity(l,r); print("no lr/uniq/speckle valid",((d>0.5)&m_).sum()/m_.sum(), "MAE", np.abs((d-gt)[m_&(d>0.5)]).mean())
m=CensusSGM(ND,uniq=0.0,speckle=0,lr_tol=1); d=m.disparity(l,r); print("lr only",((d>0.5)&m_).sum()/m_.sum())
import cv2; cv2.imwrite("scripts/_d.png", np.hstack([l, np.clip(d*3,0,255).astype(np.uint8)]))
