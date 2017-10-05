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
        row = round(0.8 * result.shape[0])  # 80% split
        print ("Amount of training data = {}".format(0.9 * result.shape[0]))
        print ("Amount of testing data = {}".format(0.1 * result.shape[0]))

        train = result[:int(row), :]  # 90% date
        x_train = train[:, :-1]  # all data until day m
        y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

        return [x_train, y_train, x_test, y_test]

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
