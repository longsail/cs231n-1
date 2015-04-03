import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def vox2col_cython(np.ndarray[DTYPE_t, ndim=5] x, int field_height,
                  int field_width, int field_depth, int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    cdef int D = x.shape[4]
    
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef int DD = (D + 2 * padding - field_depth) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=5] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width * field_depth, N * HH * WW * DD))

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    vox2col_cython_inner(cols, x_padded, N, C, H, W, D, HH, WW, DD,
                        field_height, field_width, field_depth, padding, stride)
    
    return cols


@cython.boundscheck(False)
cdef int vox2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=5] x_padded,
                             int N, int C, int H, int W, int D, int HH, int WW, int DD,
                             int field_height, int field_width, int field_depth, int padding, int stride) except? -1:
    cdef int c, ii, jj, kk, row, yy, xx, ww, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                for kk in range(field_depth):
                    row = c * field_width * field_height * field_depth + ii * field_height * field_depth + jj * field_depth + kk
                    for yy in range(HH):
                        for xx in range(WW):
                            for ww in range(DD):
                                for i in range(N):
                                    col = yy * WW * DD * N + xx * DD * N + ww * N + i
                                    cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj, stride * ww + kk]


def col2vox_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W, int D,
                  int field_height, int field_width, int field_depth, int padding, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W, D), dtype=DTYPE)
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef int DD = (D + 2 * padding - field_depth) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=5] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding, D + 2 * padding),
                                        dtype=DTYPE)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2vox_cython_inner(cols, x_padded, N, C, H, W, D, HH, WW, DD,
                        field_height, field_width, field_depth, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding, padding:-padding]
    return x_padded


@cython.boundscheck(False)
cdef int col2vox_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=5] x_padded,
                             int N, int C, int H, int W, int D, int HH, int WW, int DD,
                             int field_height, int field_width, int field_depth, int padding, int stride) except? -1:
    cdef int c, ii, jj, kk, row, yy, xx, ww, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                for kk in range(field_depth):
                    row = c * field_width * field_height * field_depth + ii * field_height * field_depth + jj * field_depth + kk
                    for yy in range(HH):
                        for xx in range(WW):
                            for ww in range(DD):
                                for i in range(N): 
                                    col = yy * WW * DD * N + xx * DD * N + ww * N + i
                                    x_padded[i, c, stride * yy + ii, stride * xx + jj, stride * ww + kk] += cols[row, col]
