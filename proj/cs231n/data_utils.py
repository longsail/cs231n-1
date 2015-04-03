import cPickle as pickle
import numpy as np
import os
import scipy.io

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'r') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):

  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def load_MODELNET10(ROOT):
  """ load all of modelnet10 """
  xs = []
  ys = []
  f = os.path.join(ROOT, 'mdb10_dim16.mat')
  # f = os.path.join(ROOT, 'testdb.mat')
  mat = scipy.io.loadmat(f)
  # print mat['testdb']
  # print mat['testdb'].shape
  # print mat['testdb'][0,0]
  X = mat['mdb'][0,0][0]
  sets = mat['mdb'][0,0][1]
  Y = mat['mdb'][0,0][2]

  mask = np.squeeze(sets == 0)
  Xtr = X[:,:,:,mask]
  Ytr = Y[mask,0]

  mask = np.squeeze(sets == 2)
  Xte = X[:,:,:,mask]
  Yte = Y[mask,0]

  Xtr = Xtr.astype('float')
  Xte = Xte.astype('float')

  return Xtr, Ytr, Xte, Yte 

  # for b in range(1,6):
  #   f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
  #   X, Y = load_CIFAR_batch(f)
  #   xs.append(X)
  #   ys.append(Y)    
  # Xtr = np.concatenate(xs)
  # Ytr = np.concatenate(ys)
  # del X, Y
  # Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  # return Xtr, Ytr, Xte, Yte