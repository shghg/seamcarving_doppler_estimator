import numpy as np

def model_frequency(t,f0,tc,dc,v,c=343):
    num=(t-tc)
    denom=np.sqrt(dc**2 + (v*(t-tc))**2)
    return f0*(1-(v/c)*(num/denom))
