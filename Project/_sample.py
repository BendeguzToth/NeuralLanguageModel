"""
In this file we can generate sample texts using a
trained model.
"""

import numpy as np
import json

from network import Network
from layers import LSTM, Dense, TimeDistributed
from losses import VectorCrossEntropy
from activations import SparseSoftmax

# The starting string to the sample.
TEXT = "F"

# The path to the weights of the trained model.
MODEL = "saves/ShakespeareNet.json"
# The path to the lookup file with which the model has been
# trained.
LOOKUP_FILE = "saves/ShakespeareLookup.json"
# Temperature of the softmax. A lower value will give more
# deterministic results. A higher value causes more variation,
# and spelling errors in the text.
TEMPERATURE = 0.5
# The length of the generated text (in number of characters).
LENGTH = 1000


with open(LOOKUP_FILE, 'r') as file:
    chars = json.load(file)

# Here we make dictionaries that can be used to convert
# between characters, integer id-s of characters, and one-hot
# vectors that will be used to represent the characters.
char_to_int = dict()
int_to_char = dict()
char_to_vec = dict()

for i in range(len(chars)):
    char_to_int[chars[i]] = i
    int_to_char[i] = chars[i]
    vec = np.zeros((len(chars), 1))
    vec[i] = 1.
    char_to_vec[chars[i]] = vec

# The length of the vector that represents a character
# is equivalent to the number of different characters
# in the text.
EMBEDDING_LENGTH = len(chars)

# Creating the model.
model = Network(
    LSTM(size=512, input_size=EMBEDDING_LENGTH, batch_size=1, backprop_depth=1, stateful=True),
    LSTM(size=512, input_size=512, batch_size=1, backprop_depth=1, stateful=True),
    TimeDistributed(Dense(size=EMBEDDING_LENGTH, input_size=512, activation=SparseSoftmax(TEMPERATURE)))
)
model.loadParams(MODEL)

# optimizer = Adam(learning_rate=lambda n: 0.001, beta_1=0.9, beta_2=0.999)

loss_function = VectorCrossEntropy

# model.assignOptimizer(optimizer)

first_input = [char_to_vec[char] for char in TEXT]
# Feed in the starting tokens.
out = model.forward(np.array([first_input]))[0, 0, :, :]
nextchar = int_to_char[np.random.choice(np.arange(len(chars)), p=out.flatten())]
TEXT += nextchar

for i in range(LENGTH-1):
    nextchar = int_to_char[np.random.choice(np.arange(len(chars)), p=model.forward([[char_to_vec[nextchar]]])[0][0].flatten())]
    TEXT += nextchar
print(TEXT)
