import os
import numpy as np
import gzip
import statistics


"""mnist dataset was manually downloaded from http://yann.lecun.com/exdb/mnist/"""
class DataLoader(object):

    def __init__(self):
        self.__path_data_dict = {}
        self.__data_dim_dict = {}
        self.__dataset_size_dict = {}

        self.__path_data_dict['mnist_train_x'] = 'train-images-idx3-ubyte.gz'
        self.__path_data_dict['mnist_train_y'] = 'train-labels-idx1-ubyte.gz'
        self.__path_data_dict['mnist_test_x'] = 't10k-images-idx3-ubyte.gz'
        self.__path_data_dict['mnist_test_y'] = 't10k-labels-idx1-ubyte.gz'

        self.__data_dim_dict['mnist'] = 28

        self.__dataset_size_dict['mnist_train'] = 60000
        self.__dataset_size_dict['mnist_test'] = 10000

    def extract_data(self, filename, num_images, dataset_name):
        img_size = self.__data_dim_dict[dataset_name]
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(img_size * img_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            data = data.reshape(num_images, img_size * img_size)
            return data

    @staticmethod
    def extract_labels(filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

    @staticmethod
    def calc_input_normalization_matrices(data):
        data_shape = data[0].shape

        average_matrix = np.ndarray(shape=data_shape)
        stdev_matrix = np.ndarray(shape=data_shape)
        print("Calculating normalization for each data...")
        print("This process can cost about 5 mins. Please hold on...")
        for i in range(data_shape[0]):
            values = [cell[i] for cell in data]
            average_matrix[i] = sum(values) / float(len(values))
            stdev_matrix[i] = statistics.stdev(values)
            print(i)
        
        return average_matrix, stdev_matrix

    @staticmethod
    def normalize_by_input(arr, average_matrix, stdev_matrix):
        data_shape = arr.shape
        new_arr = [None for i in range(data_shape[0])]

        for i in range(data_shape[0]):
            if stdev_matrix[i] == 0:
                new_arr[i] = 0
            else:
                new_arr[i] = float(arr[i] - average_matrix[i]) / stdev_matrix[i]

        return new_arr


    def load_dataset(self, name, input_normalization=True):
        #data_dir = os.path.join('data', name)
        #if not os.path.isdir(data_dir):
        #    raise Exception('dataset wasn\'t found')
        
        try:
            train_x = self.extract_data(self.__path_data_dict[name + '_train_x'],
                                        self.__dataset_size_dict[name + '_train'], name)
            train_y = self.extract_labels(self.__path_data_dict[name + '_train_y'],
                                          self.__dataset_size_dict[name + '_train'])
            test_x = self.extract_data(self.__path_data_dict[name + '_test_x'],
                                       self.__dataset_size_dict[name + '_test'], name)
            test_y = self.extract_labels(self.__path_data_dict[name + '_test_y'],
                                         self.__dataset_size_dict[name + '_test'])
            
            if input_normalization:
                # Assuming 1D input
                average_matrix, stdev_matrix = self.calc_input_normalization_matrices(train_x)
                noramalized_train_x = [self.normalize_by_input(x, average_matrix, stdev_matrix) for x in train_x]
                noramalized_test_x = [self.normalize_by_input(x, average_matrix, stdev_matrix) for x in test_x]
                return list(zip(noramalized_train_x, train_y)), list(zip(noramalized_test_x, test_y))

            train_x /= 255
            test_x /= 255
            return list(zip(train_x, train_y)), list(zip(test_x, test_y))

        except:
            raise Exception('Failed to load {} data'.format(name))
