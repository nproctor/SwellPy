import numpy as np
cimport numpy as np

def iterate (np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.int64_t, ndim=1] idx, np.ndarray[np.float64_t, ndim=3] kick, int size):
    cdef int i = 0
    while i < size:
        data[idx[i], 0] += kick[i, 0, 0]
        data[idx[i], 1] += kick[i, 0, 1]
        i += 1