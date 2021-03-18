from mxnet import nd, gpu, gluon, autograd,init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time
class Residual(nn.Block):
    def __init__(self,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),nn.BatchNorm(),
                     nn.Dense(64, activation='relu'),nn.BatchNorm(),
                     nn.Dense(64, activation='relu'),nn.BatchNorm(),
                     nn.Dense(64, activation='relu'),nn.BatchNorm(),
                     nn.Dense(64, activation='relu'),nn.BatchNorm(),
                     nn.Dense(64, activation='relu'),nn.BatchNorm(),)      
    def forward(self, X):
        return (nd.relu(self.net(X)+X))
print('Constructing network...\n')
batch_size = 64
net = nn.Sequential()
net.add(nn.Dense(64),
        Residual(), nn.BatchNorm(), nn.Dropout(0.3),
        Residual(), nn.BatchNorm(), 
        Residual(), nn.BatchNorm(), 
        Residual(), nn.BatchNorm(), 
        Residual(), nn.BatchNorm(),  
        Residual(), nn.BatchNorm(), nn.Dropout(0.3), 
        Residual(), nn.BatchNorm(), 
        Residual(), nn.BatchNorm(), 
        Residual(), nn.BatchNorm(),
        Residual(), nn.BatchNorm(), 
        nn.Dense(10))
#net.initialize(init=init.Normal())
print(net)
net.load_parameters('net.params1')
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])


mnist_train = datasets.FashionMNIST(train=True)
mnist_train = mnist_train.transform_first(transformer)
train_data = gluon.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

mnist_test = gluon.data.vision.FashionMNIST(train=False)
mnist_test = mnist_test.transform_first(transformer)
test_data = gluon.data.DataLoader(
    mnist_test, batch_size=batch_size, num_workers=0)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

print('Start training...\n')
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in test_data:
        valid_acc += acc(net(data), label)
    print("epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            valid_acc/len(test_data), time.time()-tic))
net.save_parameters('net.params1')