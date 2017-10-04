import math

from keras import callbacks
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import keras
from sklearn import metrics

import Constants
from DataProcessing import DataProcessing


class PredictionModel(object):

    def __init__(self, shape, neurons, dropout, decay):
        model = Sequential()

        model.add(LSTM(neurons[0], input_shape=(shape[0], shape[1]), return_sequences=True))
        model.add(Dropout(dropout))

        model.add(LSTM(neurons[1], input_shape=(shape[0], shape[1]), return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
        if neurons[3] == 1:
            model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
            adam = keras.optimizers.Adam(decay=decay)
            model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        else:
            model.add(Dense(neurons[3]))
            model.add(Activation('softmax'))
            model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

        model.summary()
        self.model = model

    def fit(self, X_train, y_train):
        cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        self.model.fit(
            X_train,
            y_train,
            batch_size=Constants.TRAIN_BATCH,
            callbacks=cbks,
            epochs=Constants.EPOCHS,
            validation_split=0.2,
            verbose=1,
            shuffle=True)

    def model_score(self, X_train, y_train, X_test, y_test):
        trainScore = self.model.evaluate(X_train, y_train, batch_size=Constants.TEST_BATCH, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = self.model.evaluate(X_test, y_test, batch_size=Constants.TEST_BATCH, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        return trainScore[0], testScore[0]

    def model_score_class(self, model, X_test, y_test):
        result_y = model.predict(X_test, batch_size=Constants.TEST_BATCH, verbose=1)
        result_y = DataProcessing.make_single_dimension(result_y)
        y_test = DataProcessing.make_single_dimension(y_test)
        vacc = metrics.accuracy_score(y_test, result_y)
        report = metrics.classification_report(y_test, result_y)
        print('Accuracy: %f' % vacc)
        print(report)

    def percentage_difference(self, X_test, y_test):
        percentage_diff = []

        p = self.model.predict(X_test)
        for u in range(len(y_test)):  # for each data index in test data
            pr = p[u][0]  # pr = prediction on day u

            percentage_diff.append((pr - y_test[u] / pr) * 100)
        return p
