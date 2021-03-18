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
    y0 = net[:i](X)
    y0 = y0.asnumpy()
    var.append(math.log(np.var(y0)))
print(var)
plt.plot(range(60),var)

net.initialize(force_reinit=True, init=init.Normal())
var = []
for i in range(20):
    y0 = net[:i](X)
    y0 = y0.asnumpy()
    var.append(math.log(np.var(y0)))
print(var)
plt.title('Compare')
plt.plot(range(20),var)
plt.show()