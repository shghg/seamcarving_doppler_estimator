import numpy as np

def find_seam(S, penalty=5):
    H, W = S.shape
    C = np.zeros((H, W))
    P = np.zeros((H, W), dtype=int)
    C[:,0] = S[:,0]
    for j in range(1, W):
        for i in range(H):
            best = float('inf'); best_prev=0
            for di in [-1,0,1]:
                prev=i+di
                if 0<=prev<H:
                    cost=C[prev,j-1]+S[i,j]+penalty*abs(di)
                    if cost<best:
                        best=cost; best_prev=prev
            C[i,j]=best; P[i,j]=best_prev
    seam=np.zeros(W,dtype=int)
    seam[-1]=np.argmin(C[:,-1])
    for j in range(W-1,0,-1):
        seam[j-1]=P[seam[j],j]
    return seam

def extract_if_and_amp(S,freq,time,seam):
    inst_freq=freq[seam]
    inst_amp=S[seam, np.arange(len(seam))]
    inst_amp=inst_amp/np.max(inst_amp)
    return inst_freq, inst_amp
