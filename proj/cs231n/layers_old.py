import numpy as np

def affine_forward(x, w, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
	We multiply this against a weight matrix of shape (D, M) where
	D = \prod_i d_i

	Inputs:
	x - Input data, of shape (N, d_1, ..., d_k)
	w - Weights, of shape (D, M)
	b - Biases, of shape (M,)
	
	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	out = None
	#############################################################################
	# TODO: Implement the affine forward pass. Store the result in out. You     #
	# will need to reshape the input into rows.                                 #
	#############################################################################
	N = x.shape[0]
	D = np.prod(x.shape[1:])
	x_2d = np.reshape(x,[N, D])
	out = np.dot(x_2d, w)
	out += b
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, w, b)
	return out, cache


def affine_backward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
		- x: Input data, of shape (N, d_1, ... d_k)
		- w: Weights, of shape (D, M)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	x, w, b = cache
	#############################################################################
	# TODO: Implement the affine backward pass.                                 #
	#############################################################################
	N = x.shape[0]
	D = np.prod(x.shape[1:])

	# make x two dimensional
	x_2d = np.reshape(x,[N, D])

	# backprop dout
	dx_2d = np.dot(dout, w.T)
	dw = np.dot(x_2d.T, dout)
	db = np.dot(dout.T, np.ones(N))

	# make dx the dimensions of original x
	dx = np.reshape(dx_2d,x.shape)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	#############################################################################
	# TODO: Implement the ReLU forward pass.                                    #
	#############################################################################
	out = np.maximum(0,x)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	#############################################################################
	# TODO: Implement the ReLU backward pass.                                   #
	#############################################################################
	dx = np.array(dout, copy=True)
	dx[x < 0] = 0
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and width
	W. We convolve each input with F different filters, where each filter spans
	all C channels and has height HH and width WW.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
		- 'stride': The number of pixels between adjacent receptive fields in the
			horizontal and vertical directions.
		- 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the convolutional forward pass.                           #
	# Hint: you can use the function np.pad for padding.                        #
	#############################################################################
	[N, C, H, W] = x.shape
	[F, C, HH, WW] = w.shape
	s = conv_param['stride']
	p = conv_param['pad']
	Hp = 1 + (H + 2 * p - HH) / s
	Wp = 1 + (W + 2 * p - WW) / s
	out = np.zeros([N,F,Hp,Wp])

	for n in xrange(N):
		inp = np.pad(x[n,:,:,:],p,'constant',constant_values=0)
		inp = inp[1:-1,:,:]
		for f in xrange(F):
			for wi in xrange(Wp):
				for he in xrange(Hp):
					conv = inp[:,he*s:he*s+HH,wi*s:wi*s+WW]*w[f,:,:,:]
					out[n,f,he,wi] = np.sum(conv) + b[f]
					
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, w, b, conv_param)
	return out, cache


def conv_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""
	dx, dw, db = None, None, None
	#############################################################################
	# TODO: Implement the convolutional backward pass.                          #
	#############################################################################
	x, w, b, conv_param = cache
	s = conv_param['stride']
	p = conv_param['pad']

	[N, C, H, W] = x.shape
	[F, C, HH, WW] = w.shape
	[N, F, Hp, Wp] = dout.shape

	dx = np.zeros_like(x)
	dw = np.zeros_like(w)
	db = np.zeros_like(b)

	for n in xrange(N):
		dxp = np.pad(dx[n,:,:,:],p,'constant',constant_values=0)
		dxp = dxp[1:-1,:,:]
		xp = np.pad(x[n,:,:,:],p,'constant',constant_values=0)
		xp = xp[1:-1,:,:]
		for f in xrange(F):
			for wp in xrange(Wp):
				for hp in xrange(Hp):
					dxp[:,hp*s:hp*s+HH,wp*s:wp*s+WW] += w[f,:,:,:]*dout[n,f,hp,wp]
					dw[f,:,:,:] += xp[:,hp*s:hp*s+HH,wp*s:wp*s+WW]*dout[n,f,hp,wp]
					db[f] += dout[n,f,hp,wp]
		dx[n,:,:,:] = dxp[:,1:-1,1:-1]

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx, dw, db

def conv3_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and width
	W and depth D. We convolve each input with F different filters, where each 
	filter spans all C channels and has height HH, width WW and depth DD.

	Input:
	- x: Input data of shape (N, C, H, W, D)
	- w: Filter weights of shape (F, C, HH, WW, D)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
		- 'stride': The number of pixels between adjacent receptive fields in the
			horizontal and vertical directions.
		- 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W', D') where H', W' and D' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
		D' = 1 + (D + 2 * pad - DD) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the convolutional forward pass.                           #
	# Hint: you can use the function np.pad for padding.                        #
	#############################################################################
	[N, C, H, W, D] = x.shape
	[F, C, HH, WW, DD] = w.shape
	s = conv_param['stride']
	p = conv_param['pad']
	Hp = 1 + (H + 2 * p - HH) / s
	Wp = 1 + (W + 2 * p - WW) / s
	Dp = 1 + (D + 2 * p - DD) / s
	out = np.zeros([N,F,Hp,Wp,Dp])

	for n in xrange(N):
		inp = np.pad(x[n,:,:,:,:],p,'constant',constant_values=0)
		inp = inp[1:-1,:,:,:]
		for f in xrange(F):
			for hp in xrange(Hp):
				for wp in xrange(Wp):
					for dp in xrange(Dp):
						conv = inp[:,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD]*w[f,:,:,:,:]
						out[n,f,hp,wp,dp] = np.sum(conv) + b[f]
					
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, w, b, conv_param)
	return out, cache

def conv3_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""
	dx, dw, db = None, None, None
	#############################################################################
	# TODO: Implement the convolutional backward pass.                          #
	#############################################################################
	x, w, b, conv_param = cache
	s = conv_param['stride']
	p = conv_param['pad']

	[N, C, H, W, D] = x.shape
	[F, C, HH, WW, DD] = w.shape
	[N, F, Hp, Wp, Dp] = dout.shape

	dx = np.zeros_like(x)
	dw = np.zeros_like(w)
	db = np.zeros_like(b)

	for n in xrange(N):
		dxp = np.pad(dx[n,:,:,:,:],p,'constant',constant_values=0)
		dxp = dxp[1:-1,:,:,:]
		xp = np.pad(x[n,:,:,:,:],p,'constant',constant_values=0)
		xp = xp[1:-1,:,:,:]
		for f in xrange(F):
			for hp in xrange(Hp):
				for wp in xrange(Wp):
					for dp in xrange(Dp):
						dxp[:,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD] += w[f,:,:,:,:]*dout[n,f,hp,wp,dp]
						dw[f,:,:,:,:] += xp[:,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD]*dout[n,f,hp,wp,dp]
						db[f] += dout[n,f,hp,wp,dp]
		dx[n,:,:,:,:] = dxp[:,1:-1,1:-1,1:-1]

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx, dw, db


def max_pool_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
		- 'pool_height': The height of each pooling region
		- 'pool_width': The width of each pooling region
		- 'stride': The distance between adjacent pooling regions

	Returns a tuple of:
	- out: Output data
	- cache: (x, pool_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the max pooling forward pass                              #
	#############################################################################
	[N, C, H, W] = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	s = pool_param['stride']
	Hp = 1 + (H - HH) / s
	Wp = 1 + (W - WW) / s

	out = np.zeros([N,C,Hp,Wp])

	for n in xrange(N):
		for wp in xrange(Wp):
			for hp in xrange(Hp):
				poolme = x[n,:,hp*s:hp*s+HH,wp*s:wp*s+WW]
				out[n,:,hp,wp] = np.max(poolme.reshape([C,HH*WW]),axis=1)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, pool_param)
	return out, cache


def max_pool_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	#############################################################################
	# TODO: Implement the max pooling backward pass                             #
	#############################################################################
	x, pool_param = cache
	[N, C, H, W] = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	s = pool_param['stride']
	Hp = 1 + (H - HH) / s
	Wp = 1 + (W - WW) / s

	dx = np.zeros_like(x)

	for n in xrange(N):
		for c in xrange(C):
			for wp in xrange(Wp):
				for hp in xrange(Hp):
					poolme = x[n,c,hp*s:hp*s+HH,wp*s:wp*s+WW]
					poolme_flat = np.reshape(poolme,[HH*WW])
					poolme_out = np.zeros_like(poolme_flat)

					poolme_out[np.argmax(poolme_flat)] = 1
					dx[n,c,hp*s:hp*s+HH,wp*s:wp*s+WW] = np.reshape(poolme_out,[HH,WW]) * dout[n,c,hp,wp]


	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def max_pool3_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
		- 'pool_height': The height of each pooling region
		- 'pool_width': The width of each pooling region
		- 'stride': The distance between adjacent pooling regions

	Returns a tuple of:
	- out: Output data
	- cache: (x, pool_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the max pooling forward pass                              #
	#############################################################################
	[N, C, H, W, D] = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	DD = pool_param['pool_depth']
	s = pool_param['stride']
	Hp = 1 + (H - HH) / s
	Wp = 1 + (W - WW) / s
	Dp = 1 + (D - DD) / s

	out = np.zeros([N,C,Hp,Wp,Dp])

	for n in xrange(N):
		for hp in xrange(Hp):
			for wp in xrange(Wp):
				for dp in xrange(Dp):
					poolme = x[n,:,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD]
					out[n,:,hp,wp,dp] = np.max(poolme.reshape([C,HH*WW*DD]),axis=1)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, pool_param)
	return out, cache


def max_pool3_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	#############################################################################
	# TODO: Implement the max pooling backward pass                             #
	#############################################################################
	x, pool_param = cache
	[N, C, H, W, D] = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	DD = pool_param['pool_depth']
	s = pool_param['stride']
	Hp = 1 + (H - HH) / s
	Wp = 1 + (W - WW) / s
	Dp = 1 + (D - DD) / s

	dx = np.zeros_like(x)

	for n in xrange(N):
		for c in xrange(C):
			for hp in xrange(Hp):
				for wp in xrange(Wp):
					for dp in xrange(Dp):
						poolme = x[n,c,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD]
						poolme_flat = np.reshape(poolme,[HH*WW*DD])
						poolme_out = np.zeros_like(poolme_flat)

						poolme_out[np.argmax(poolme_flat)] = 1
						dx[n,c,hp*s:hp*s+HH,wp*s:wp*s+WW,dp*s:dp*s+DD] = np.reshape(poolme_out,[HH,WW,DD]) * dout[n,c,hp,wp,dp]


	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def svm_loss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	dx /= N
	return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	probs = np.exp(x - np.max(x, axis=1, keepdims=True))
	probs /= np.sum(probs, axis=1, keepdims=True)
	N = x.shape[0]
	loss = -np.sum(np.log(probs[np.arange(N), y])) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx

