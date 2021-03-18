import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss, utils as gutils
import numpy as np
import math
import matplotlib.pyplot as plt

layer_num = 60
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False))
X = nd.random.uniform(-1,1,(1,100))
net.initialize(force_reinit=True, init=init.Xavier())
var = []
for i in range(60):
    layer_output = net[:i](X)
    layer_output.attach_grad()
    with autograd.record():
        Y = net[i:](layer_output)
    Y.backward()
    y0 = layer_output.grad
    y0 = y0.asnumpy()
    v = math.log(np.var(y0))
    var.append(v)
plt.plot(range(60),var)

net.initialize(force_reinit=True, init=init.Normal())
var = []
for i in range(20):
    i = i + 40
    layer_output = net[:i](X)
    layer_output.attach_grad()
    with autograd.record():
        Y = net[i:](layer_output)
    Y.backward()
    y0 = layer_output.grad
    y0 = y0.asnumpy()
    print(np.var(y0))
    v = math.log(np.var(y0))
    var.append(v)

plt.plot(range(40,60),var)
plt.title('Compare')
plt.show()
