import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss, utils as gutils
import numpy as np
import math
import matplotlib.pyplot as plt

layer_num = 500
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False))    
    #net.add(nn.BatchNorm())
net.initialize(force_reinit=True, init=init.Xavier())
X = nd.random.uniform(-1,1,(32, 100))

var = []
for i in range(500):
    layer_output = net[:i](X)
    layer_output.attach_grad()
    with autograd.record():
        Y = net[i:](layer_output)
    Y.backward()
    y0 = layer_output.grad
    y0 = y0.asnumpy()
    v = math.log(np.var(y0))
    var.append(v)
plt.plot(range(500),var,label = 'Without BN')


layer_num = 500
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False))    
    net.add(nn.BatchNorm())
net.initialize(force_reinit=True, init=init.Xavier())
X = nd.random.uniform(-1,1,(32, 100))

var = []
for i in range(500):
    layer_output = net[:i](X)
    layer_output.attach_grad()
    with autograd.record():
        Y = net[i:](layer_output)
    Y.backward()
    y0 = layer_output.grad
    y0 = y0.asnumpy()
    v = math.log(np.var(y0))
    var.append(v)
plt.plot(range(500),var,label = 'With BN')
plt.title('Compare')
plt.legend(loc='upper right')
plt.show()
