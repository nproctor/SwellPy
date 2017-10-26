cimport cython
import numpy as np
cimport numpy as np

def iterate (double[:, :] data, long[:] idx, double[:, :, :] kick, long size):
    cdef long i, N = size
    for i in range(N):
        data[idx[i], 0] += kick[i, 0, 0]
        data[idx[i], 1] += kick[i, 0, 1]