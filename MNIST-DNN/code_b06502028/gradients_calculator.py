import numpy as np
from utils import ACTIVATION_NAME_TO_DERIVATIVE, ACTIVATION_NAME_TO_FUNC


class GradientsCalculator(object):
    def __init__(self, activation_name):
        self.__activation_function = ACTIVATION_NAME_TO_FUNC[activation_name]
        self.__activation_derivative = ACTIVATION_NAME_TO_DERIVATIVE[activation_name]

    def calc_gradients(self, x, y, y_pred, params):
        gradients_values = []
        y_label = np.zeros(params[-1][1].shape)
        y_label[y] = 1

        z_l, a_l = self.__calc_params_for_gradients(x, params)
        g_last_b = y_pred - y_label

        gradients_values.insert(0, (GradientsCalculator.__calc_weight_matrix_grad(a_l[-2], g_last_b), g_last_b))

        i = len(params) - 2
        while i > -1:
            gb = self.__calc_weight_vector_grad(gradients_values[0][1], params[i + 1][0], a_l[i + 1])
            gw = GradientsCalculator.__calc_weight_matrix_grad(z_l[i], gb)
            gradients_values.insert(0, [gw, gb])
            i -= 1
        return gradients_values

    @staticmethod
    def __calc_weight_matrix_grad(x, gb):
        return np.array([x]).transpose().dot([gb])

    def __calc_weight_vector_grad(self, grad, mat, a):
        return grad.dot(mat.transpose()) * self.__activation_derivative(a)

    def __calc_params_for_gradients(self, x, params):
        z_l = [x]
        a_l = [x]

        i = 0
        while i < len(params):
            z_l.append(np.array(z_l[-1]).dot(params[i][0]) + params[i][1])
            a_l.append(self.__activation_function(z_l[-1]))
            i += 1
        return z_l, a_l
