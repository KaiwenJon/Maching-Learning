import matplotlib.pyplot as plt
import numpy as np
from mxnet import nd, gluon, init, autograd
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss, utils as gutils

net = nn.Sequential()
net.add(nn.Dense(200, activation='tanh', use_bias=False),
        nn.Dense(200, activation='tanh', use_bias=False),
        nn.Dense(200, activation='tanh', use_bias=False),
        nn.Dense(200, activation='tanh', use_bias=False)
       )

X = nd.random.uniform(-1,1,(1,100))
net.initialize(force_reinit=True, init=init.Xavier())
X.attach_grad()
with autograd.record():
    Y = net[:1](X)
Y.backward()
y0 = X.grad
y0 = y0.asnumpy()
y0 = np.round(y0,1)
y0 = list(y0[0])
#print(y0)
y0 = sorted(y0)
print(y0)
y0set = sorted(set(y0))
print(y0set)
n = 0;
#plt.figure()
x = []
y = []
for item in y0set:
    x.append(item)
    y.append(y0.count(item))
    n += 1
#print(x)
#print(y)
plt.plot(x,y,label = 'layer1')

with autograd.record():
    Y = net[:2](X)
Y.backward()
y0 = X.grad
y0 = y0.asnumpy()
y0 = np.round(y0,1)
y0 = list(y0[0])
#print(y0)
y0 = sorted(y0)
print(y0)
y0set = sorted(set(y0))
print(y0set)
n = 0;
#plt.figure()
x = []
y = []
for item in y0set:
    x.append(item)
    y.append(y0.count(item))
    n += 1
#print(x)
#print(y)
plt.plot(x,y,label = 'layer2')

with autograd.record():
    Y = net[:3](X)
Y.backward()
y0 = X.grad
y0 = y0.asnumpy()
y0 = np.round(y0,1)
y0 = list(y0[0])
#print(y0)
y0 = sorted(y0)
print(y0)
y0set = sorted(set(y0))
print(y0set)
n = 0;
#plt.figure()
x = []
y = []
for item in y0set:
    x.append(item)
    y.append(y0.count(item))
    n += 1
#print(x)
#print(y)
plt.plot(x,y,label = 'layer3')

with autograd.record():
    Y = net[:4](X)
Y.backward()
y0 = X.grad
y0 = y0.asnumpy()
y0 = np.round(y0,1)
y0 = list(y0[0])
#print(y0)
y0 = sorted(y0)
print(y0)
y0set = sorted(set(y0))
print(y0set)
n = 0;
#plt.figure()
x = []
y = []
for item in y0set:
    x.append(item)
    y.append(y0.count(item))
    n += 1
#print(x)
#print(y)
plt.plot(x,y,label = 'layer4')
plt.legend(loc='upper right')


plt.show()