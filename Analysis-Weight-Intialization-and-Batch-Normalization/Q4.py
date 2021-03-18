import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss, utils as gutils
import numpy as np
import math
import matplotlib.pyplot as plt

layer_num = 100
net = nn.Sequential()
for i in range(layer_num):
    net.add(nn.Dense(100, activation="tanh", use_bias=False))    
    net.add(nn.BatchNorm())
net.initialize(force_reinit=True, init=init.Normal())
X = nd.random.uniform(-1,1,(128, 100))

var = []
for i in range(20):
    y0 = net[:i](X)
    y0 = y0.asnumpy()
    v = math.log(np.var(y0))
    var.append(v)
print(var)
plt.plot(range(20),var,label = 'Normal')

plt.show()
