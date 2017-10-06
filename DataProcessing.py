import numpy as np


class DataProcessing(object):
    @staticmethod
    def load_data(stock, seq_len):
        amount_of_features = len(stock.columns)
        print ("Amount of features = {}".format(amount_of_features))
        data = stock.as_matrix()
        sequence_length = seq_len + 1  # index starting from 0
        result = []

        for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
            result.append(data[index: index + sequence_length])  # index : index + 22days

        result = np.array(result)
        print ("Amount of data = {}".format(result.shape[0]))

        x = result[:, :-1]  # all data until day m
        y = result[:, -1][:, -1]  # day m + 1 adjusted close price

        x_train = np.reshape(x, (x.shape[0], x.shape[1], amount_of_features))
        return [x_train, y]

    @staticmethod
    def make_single_dimension(y):
        y = np.copy(y)
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_value = np.argmax(y, axis=1)
        else:
            y[y > 0.5] = 1
            y[y < 0.5] = 0
            y_value = y.astype(int)
            if len(y.shape) > 1 and y.shape[1] == 1:
                y_value = y_value[:, 0]

        return y_value
