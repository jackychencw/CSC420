import numpy as np 

u = np.array([[1/4,1/2,3/4,1,3/4,1/2,1/4]])
v = u.T
print u.shape
r = np.dot(v, u)
print(r)