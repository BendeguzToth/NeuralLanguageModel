"""
This file is used to train the network.
"""

import numpy as np
import json
import time

from network import Network
from layers import LSTM, Dense, TimeDistributed
from losses import VectorCrossEntropy
from activations import SparseSoftmax
from optimizers import RMSprop


SEQUENCE_LENGTH = 100
BATCH_SIZE = 25
NR_OF_EPOCHS = 100

# Path to the source text file
SOURCE_FILE = "saves/shakespeare.txt"
# Path to the lookup file.
LOOKUP_FILE = "saves/ShakespeareLookup.json"
# The model will be saved with this name.
MODEL_NAME = "ShakespeareNet"
# Path to the directory the model will be saved on.
# Needs to end with '/'!
MODEL_PATH = "saves/"

# In case you want to train a saved network,
# fill in the path to the weight file.
# If you wish to train a new model, use an empty string.
RESTORE_MODEL_PATH = "saves/ShakespeareNet.json"
# Path to saved optimizer, if you wish to resume training.
# Use an empty string to initialize a new optimizer.
RESTORE_OPTIMIZER_PATH = "saves/ShakespeareOptimizer.json"
# When resuming training, set the epoch counter to
# where you left off.
INITIAL_EPOCH = 8


def makeBatches(data, batch_size, embedding_length):
    """
    Generator to produce sequences to train from.
    :param data: np.array of characters.
    :param batch_size: The size of the batch.
    :param embedding_length: The length of the input
    vectors.
    """
    def vectorize(data_):
        """
        This function converts a character array to
        an array of one-hot representations.
        """
        buff = np.zeros(shape=(BATCH_SIZE, SEQUENCE_LENGTH + 1, embedding_length, 1))
        for batch_idx in range(len(data_)):
            for char in range(len(data_[batch_idx])):
                buff[batch_idx, char] = char_to_vec[data_[batch_idx, char]]
        return buff
    begin = time.time()
    for i in range(1, data.shape[1] - batch_size + 1, batch_size):
        cache = vectorize(data[:, i-1:i+batch_size])
        print(f"{(i-1)//batch_size+1}/{data.shape[1]//batch_size}")
        end = time.time()
        print(f"Time taken: {end - begin}")
        begin = end
        yield (cache[:, :-1, :], cache[:, 1:, :])


# Loading in the file.
with open(SOURCE_FILE, 'r', encoding='utf8') as file:
    source = file.read()

try:
    with open(LOOKUP_FILE, 'r') as file:
        chars = json.load(file)
except FileNotFoundError:
    chars = list(set(source))
    with open(LOOKUP_FILE, 'w') as file:
        json.dump(chars, file)

char_to_int = dict()
int_to_char = dict()
char_to_vec = dict()

for i in range(len(chars)):
    char_to_int[chars[i]] = i
    int_to_char[i] = chars[i]
    vec = np.zeros((len(chars), 1))
    vec[i] = 1.
    char_to_vec[chars[i]] = vec

source = np.array(list(source))[:(len(source) // BATCH_SIZE) * BATCH_SIZE]
source = np.array(np.split(source, BATCH_SIZE))

EMBEDDING_LENGTH = len(chars)

# Creating the model.
model = Network(
    LSTM(size=512, input_size=EMBEDDING_LENGTH, batch_size=BATCH_SIZE, backprop_depth=SEQUENCE_LENGTH, stateful=True),
    LSTM(size=512, input_size=512, batch_size=BATCH_SIZE, backprop_depth=SEQUENCE_LENGTH, stateful=True),
    TimeDistributed(Dense(size=EMBEDDING_LENGTH, input_size=512, activation=SparseSoftmax()))
)

if RESTORE_MODEL_PATH:
    model.loadParams(RESTORE_MODEL_PATH)


optimizer = RMSprop(learning_rate=lambda n: 0.001)
loss_function = VectorCrossEntropy

model.assignOptimizer(optimizer)

if RESTORE_OPTIMIZER_PATH:
    optimizer.load(RESTORE_OPTIMIZER_PATH)


for epoch in range(INITIAL_EPOCH, NR_OF_EPOCHS + INITIAL_EPOCH):
    loss, accuracy = model.train(makeBatches(source, SEQUENCE_LENGTH, EMBEDDING_LENGTH), lossfunc=loss_function)
    model.saveParams(f"{MODEL_PATH}{MODEL_NAME}-{epoch:02d}-loss_{loss:.5f}-acc_{accuracy:.5f}.nn")
    optimizer.save(f"{MODEL_PATH}{epoch:02d}-optimizer.json")
    print(f"Epoch: {epoch} - loss: {loss} - accuracy: {accuracy}")
