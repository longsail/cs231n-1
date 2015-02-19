import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_width) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.tile(np.repeat(np.arange(field_height), field_width), C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

  #############################################################################
  #                               3D STUFFS                                   #
  #############################################################################

def get_vox2col_indices(x_shape, field_height, field_width, field_depth, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W, D = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_width) % stride == 0
  assert (D + 2 * padding - field_depth) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1
  out_depth = (D + 2 * padding - field_depth) / stride + 1

  i0 = np.arange(field_height)
  i0 = np.repeat(i0, field_width*field_depth)
  i0 = np.tile(i0, C)

  i1 = np.arange(out_height)
  i1 = stride * np.repeat(i1, out_width*out_depth)

  j0 = np.arange(field_width)
  j0 = np.repeat(j0, field_depth)
  j0 = np.tile(j0, field_height)
  j0 = np.tile(j0, C)

  j1 = np.arange(out_width)
  j1 = np.repeat(j1, out_depth)
  j1 = stride * np.tile(j1, out_height)

  k0 = np.arange(field_depth)
  k0 = np.tile(k0, field_width * field_height * C)
  k1 = stride * np.tile(np.arange(out_depth), out_height*out_width)

  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = k0.reshape(-1, 1) + k1.reshape(1, -1)

  # print 'i'
  # print i0.reshape(-1, 1),i1.reshape(1, -1)
  # print i
  # print 'j'
  # print j0.reshape(-1, 1),j1.reshape(1, -1)
  # print j
  # print 'k'
  # print k0.reshape(-1, 1),k1.reshape(1, -1)
  # print k

  l = np.repeat(np.arange(C), field_height * field_width * field_depth).reshape(-1, 1)
  # print 'l'
  # print l

  return (l, i, j, k)


def vox2col_indices(x, field_height, field_width, field_depth, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p), (p, p)), mode='constant')

  l, i, j, k = get_vox2col_indices(x.shape, field_height, field_width, field_depth, 
                 padding, stride)

  cols = x_padded[:, l, i, j, k]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height*field_width*field_depth * C, -1)
  return cols

pass
