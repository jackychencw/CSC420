import numpy as np

def isSeparable(m):
    u, s, vh = np.linalg.svd(m)
    vs = np.diag(s)
    threshold = 1e-7
    non_vanishing_sv = vs[vs > threshold]
    if len(non_vanishing_sv) == 1:
        vf = np.sqrt(s[0]) * u[:,0]
        hf = np.sqrt(s[0]) * vh[0]
        return True, vf, hf
    else:
        return False, None, None

A = np.array([[1,1,1],[1,1,1],[1,1,1]])
print isSeparable(A)

B = np.array([[0,1,1],[1,1,1],[1,1,1]])
print isSeparable(B)
    
