import numpy as np
from utils import calc_init_fcnn_values, softmax, ACTIVATION_NAMES
from gradients_calculator import GradientsCalculator
import random
from utils import ACTIVATION_NAME_TO_FUNC
import math


class FCClassifierModel(object):
    """We'll use the same activation function for the entire network sigmoid/tanh/relu"""
    def __init__(self, dim_sizes, activation='relu'):
        if activation not in ACTIVATION_NAMES:
            raise Exception('Unknown activation function')
        self.__activation_function = ACTIVATION_NAME_TO_FUNC[activation]
        self.__dims = dim_sizes
        self.__params = self.__init_params(dim_sizes)
        self.__activation_name = activation
        self.__gradient_calculator = GradientsCalculator(activation)

    def __init_params(self, dims):
        params = []
        init_value = calc_init_fcnn_values(dims[0], dims[-1])
        for index, val in enumerate(dims):
            if (index != len(dims) - 1):
                params.append([np.random.uniform(-init_value, init_value, (dims[index], dims[index + 1])),
                               np.zeros(dims[index + 1])])
        return params

    def __classifier_output(self, x):
        i = 0
        value = x
        while i < len(self.__params) - 1:
            value = self.__activation_function(np.dot(value, self.__params[i][0]) + self.__params[i][1])
            i += 1
        return softmax(np.dot(value, self.__params[-1][0]) + self.__params[-1][1])

    def predict(self, x):
        return np.argmax(self.__classifier_output(x))

    def calc_loss(self, y, y_pred):
        return -np.log(y_pred[y])

    @staticmethod
    def __zero_params(params):
        for wb in params:
            wb[0].fill(0.0)
            wb[1].fill(0.0)

    @staticmethod
    def add_params_gradients(LR, params, grads):
        i = 0
        while i < len(params):
            params[i][0] -= LR * grads[i][0]
            params[i][1] -= LR * grads[i][1]
            i += 1

    def train(self, train_set, epochs, val=None, use_minibatch=True, minibatch_size=20, LR=0.003, use_decreasing_lr=True):
        print('Start training.')
        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        for epoch in range(epochs):
            print('Calculating Epoch {}/{}...'.format(epoch + 1, epochs))
            if use_decreasing_lr and epoch > 15:
                lr = LR * math.pow(0.98, epochs)
                if lr < 0.0001:
                    lr = 0.0001
            else:
                lr = LR
            output = []
            #output.append('Epoch {}/{}:'.format(epoch + 1, epochs))
            train_loss = 0.0
            random.shuffle(train_set)
            if use_minibatch:
                k = 0
                mini_batch_params = self.__init_params(self.__dims)
                self.__zero_params(mini_batch_params)

            for x, y in train_set:
                y_pred = self.__classifier_output(x)
                loss = self.calc_loss(y, y_pred)
                train_loss += loss

                grads = self.__gradient_calculator.calc_gradients(x, y, y_pred, self.__params)

                if use_minibatch:
                    FCClassifierModel.add_params_gradients(lr, mini_batch_params, grads)
                    k += 1
                    if ((k != 0 and k % minibatch_size == 0) or k == len(train_set) - 1):
                        for wb, wb_grads in zip(self.__params, mini_batch_params):
                            wb[0] += wb_grads[0]
                            wb[1] += wb_grads[1]
                        self.__zero_params(mini_batch_params)
                else:
                    FCClassifierModel.add_params_gradients(lr, self.__params, grads)

            train_loss /= len(train_set)
            train_accuracy, _ = self.__calc_accuracy_and_loss(train_set)

            train_acc_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            output.append('Accuracy on training data {}'.format(train_accuracy))
            output.append('Train loss {}'.format(train_loss))

            if val is not None:
                val_accuracy, val_loss = self.__calc_accuracy_and_loss(val)
                val_acc_list.append(val_accuracy)
                val_loss_list.append(val_loss)
                output.append('Validation accuracy {}'.format(val_accuracy))
                output.append('Validation loss {}'.format(val_loss))
            output.append('\n')

            for t in output:
                print(t)

    def test(self, test_set):
        acc, loss = self.__calc_accuracy_and_loss(test_set)
        print('Accuracy on testing data: {}, test loss: {}'.format(acc, loss))
        print("\n")

    def __calc_accuracy_and_loss(self, data_set):
        good_preds, bad_preds, loss = 0.0, 0.0, 0.0
        for x, y in data_set:
            y_pred = self.__classifier_output(x)
            loss += self.calc_loss(y, y_pred)
            if np.argmax(y_pred) == y:
                good_preds += 1
            else:
                bad_preds += 1
        return good_preds / (good_preds + bad_preds), loss / len(data_set)
