import pandas
import logging
import datetime
from util.ConfigBasedProgram import TimedProgram, ConfigBasedProgram
from keras.models import Sequential
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


class NNClassifier(TimedProgram):

    def execute(self):

        # hidden_layers = self.get_config("hidden_layers")
        # activation_fn = self.get_config("activation_fn")
        # optimizer = self.get_config("optimizer")
        # loss_fn = self.get_config("loss_fn")

        data_file = self.get_config("data_file")
        class_column = self.get_config("class_column")
        drop_columns = self.get_config("drop_columns")
        dtype = self.get_config("dtype")



        model = Sequential()
        model.add(Dense(512, input_shape=()))
        model.add(Dense(100, input_shape=))






if __name__ == "__main__":
    prog = ConfigBasedProgram("Use NN to classify reviews", TimedProgram)
    prog.main()
