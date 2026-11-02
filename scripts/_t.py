import sys; sys.argv=['x']
from scripts.classical_bench import *
pairs=list(gt_pairs())
for name,m in {"tol15 p1 40 p2 500 e150":CensusSGM(ND,uniq=0.0,tol=15,p1=40,p2=500,p2_edge=150),"tol20 p1 40 p2 600 e200":CensusSGM(ND,uniq=0.0,tol=20,p1=40,p2=600,p2_edge=200),"tol20 p1 30 p2 300 e100":CensusSGM(ND,uniq=0.0,tol=20,p1=30,p2=300,p2_edge=100),"4paths tol15 p1 30 p2 300":CensusSGM(ND,uniq=0.0,tol=15,p1=30,p2=300,p2_edge=100,paths=4)}.items():
    v,e,t=[],[],[]
    for l,r,m_,gt,_ in pairs:
        t0=time.perf_counter(); d=m.disparity(l,r); t.append(time.perf_counter()-t0); ok=m_&(d>0.5); v.append(ok.sum()/m_.sum()); e.append((d-gt)[ok])
    e=np.concatenate(e); g=np.abs(e)<3
    print(f"{name:18s} {np.median(t)*1000:5.0f}ms valid {np.mean(v)*100:5.1f} MAE {np.abs(e[g]).mean():.3f} out>1 {(np.abs(e)>1).mean()*100:.1f} bias {np.median(e):.3f}")
