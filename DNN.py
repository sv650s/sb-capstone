import pandas
import logging
import datetime
from util.program_util import TimedProgram, ConfigFileBasedProgram
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.layers import Dense, Activation

"""
From this doc:

https://www.heatonresearch.com/2017/06/01/hidden-layers.html

Number of hidden layers:

none	Only capable of representing linear separable functions or decisions.
1	Can approximate any function that contains a continuous mapping from one finite space to another.
2	Can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy.
>2	Additional layers can learn complex representations (sort of automatic feature engineering) for layer layers.

Number of hidden neurons:

The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.

"""

log = logging.getLogger(__name__)


class DNN(object):

    def __init__(self,
                 num_input_features: int,
                 batch_size=128,
                 epoch: int = 5,
                 verbose=1):

        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

        self.early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

        model = Sequential()

        model.add(Dense(128, input_shape=(num_input_features,), kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128, kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(5, activation='relu'))
        model.add(Activation('softmax'))
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        # use for fitting later
        self.network_history = None

    def save(self, filename):
        self.model.save(filepath=filename, overwrite=True, include_optimizer=True)

    def load(self, filename):
        self.model = load_model(filename)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def fit(self, x_train, y_train):
        y_train = OneHotEncoder().fit_transform(y_train.values.reshape(len(y_train), 1)).toarray()

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=1)

        self.network_history = self.model.fit(x_train, y_train,
                                              batch_size=self.batch_size,
                                              epochs=self.epoch,
                                              verbose=self.verbose,
                                              validation_data=(x_val, y_val),
                                              callbacks=[self.early_stop])
