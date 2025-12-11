import numpy as np
from .model import model_frequency

def sse(params,t,f_meas,c=343):
    f0,tc,dc,v=params
    f_pred=model_frequency(t,f0,tc,dc,v,c)
    return np.sum((f_meas-f_pred)**2)

def coarse_search(t,f_meas):
    v_candidates=np.linspace(1,40,50)
    tc_candidates=np.linspace(min(t),max(t),200)
    f0_init=np.mean(f_meas); dc_init=10
    best=None; best_err=float('inf')
    for v in v_candidates:
        for tc in tc_candidates:
            err=sse([f0_init,tc,dc_init,v],t,f_meas)
            if err<best_err:
                best_err=err; best=(f0_init,tc,dc_init,v)
    return best

def refine_params(initial,t,f_meas,max_iter=20,lr=1e-5):
    params=np.array(initial,float)
    eps=1e-3
    for _ in range(max_iter):
        grad=np.zeros(4)
        for i in range(4):
            p1=params.copy(); p2=params.copy()
            p1[i]+=eps; p2[i]-=eps
            grad[i]=(sse(p1,t,f_meas)-sse(p2,t,f_meas))/(2*eps)
        params-=lr*grad
    return params
