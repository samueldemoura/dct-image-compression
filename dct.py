import math
import matplotlib.pyplot as plt

def dct(x, inverse=False):
    N = len(x)
    X = [0]*N

    for k in range(0, N):
        if k == 0:
            ck = math.sqrt(0.5)
        else:
            ck = 1

        _sum = 0
        for n in range(0, N):
            theta = 2*math.pi * (k / (2*N)) * n + (k*math.pi) / (2*N)
            _sum += x[n] * math.cos(theta)
            if inverse:
                _sum *= ck
        
        X[k] = math.sqrt(2/N) * _sum
        if not inverse:
            X[k] *= ck

    return X

# Testing
x = [0] * 26
for i in range(0, 26):
    x[i] =  math.cos(2*math.pi*10*i * (math.pi / 180)) #math.cos(100*i * (math.pi / 180))

X = dct(x)
plt.plot(x)
plt.plot(X)
plt.show()