#!/usr/bin/env python3

"""
Here we test how to use multiple GPUs with cunumeric
"""

import numpy as np
from time import time


N = 2000

print(f"Size of the matrices = {N}")
f = open("times-cunumeric.dat", 'w')

A = np.random.random([N, N])
B = np.random.random([N, N])
C = np.random.random([N, N])


"""
# following is equivalent to matrix multiplication
for i in range(N):
    term = 0
    for j in range(N):
        for k in range(N):
            for l in range(N):
                term[j, l] += B[j, k] * C[k, l]
        res[i, l] += A[i, j] * term
"""


# and is done by simply
t1 = time()
res = np.einsum('ij,jk,kl->il', A, B, C)
t2 = time()
print(nd, t2 - t1)
f.write(f"{t2 - t1}\n")
f.flush()
#
f.close()