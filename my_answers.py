import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # reshape each 
    for i in range(0, len(series) - window_size):
        output_index = i + window_size
        X.append(series[i:output_index])
        y.append(series[output_index])
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    lstm_cell = LSTM(5, dropout=0.0, stateful=False, input_shape=(window_size, 1))
    model.add(lstm_cell)
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = set(['!', ',', '.', ':', ';', '?'])
    desired_chars = punctuation.union(set(string.ascii_lowercase))
    all_chars = set(text)
    undesired_chars = all_chars - desired_chars

    for _, undesired_char in enumerate(undesired_chars):
        text = text.replace(undesired_char, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # iterate through and add to inputs and outputs
    # use step_size to iterate
    for i in range(0, len(text) - window_size, step_size):
        t_input = text[i:i+window_size]
        inputs.append(t_input)
        t_output = text[i+window_size]
        outputs.append(t_output)

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    lstm = LSTM(200, input_shape=(window_size, num_chars), dropout=0.5)
    model.add(lstm)
    model.add(Dense(num_chars, input_shape=(200,), activation='softmax'))
    return model
