import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss, utils as gutils
import numpy as np
import math
import matplotlib.pyplot as plt

layer_num = 500
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False),
            nn.BatchNorm())
net.initialize(force_reinit=True, init=init.Xavier())
X = nd.random.uniform(-1,1,(32, 100))
var = []
for i in range(500):
    y0 = net[:i](X)
    y0 = y0.asnumpy()
    var.append(math.log(np.var(y0)))
print(var)
plt.plot(range(500),var,label = 'With BN')

layer_num = 500
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False))
net.initialize(force_reinit=True, init=init.Xavier())
X = nd.random.uniform(-1,1,(32, 100))
var = []
for i in range(500):
    y0 = net[:i](X)
    y0 = y0.asnumpy()
    var.append(math.log(np.var(y0)))
print(var)
plt.plot(range(500),var,label = 'Without BN')
plt.legend(loc='upper right')
plt.show()