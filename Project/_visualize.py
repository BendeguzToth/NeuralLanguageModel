"""
In this file we visualize the activations of
particular neurons, at different positions
of a provided sample text.
"""

# Standard libraries
import json
import tkinter as tk

# Third-party libraries
import numpy as np

# Project files
from layers import LSTM

# SETUP
MODEL = "saves/ShakespeareNet.json"
LOOKUP_FILE = "saves/ShakespeareLookup.json"
TEXT_FILE = "saves/sample.txt"


def main():
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
    # Create the LSTM layers only. We don't use the Network class,
    # since we are only interested in the activations of the recurrent
    # layers.
    first_layer = LSTM(size=512, input_size=EMBEDDING_LENGTH, batch_size=1, backprop_depth=1, stateful=True)
    second_layer = LSTM(size=512, input_size=512, batch_size=1, backprop_depth=1, stateful=True)

    # Load the weights.
    with open(MODEL, 'r') as file:
        weights = json.load(file)
    first_layer.loadParams(weights[0])
    second_layer.loadParams(weights[1])

    # Loading in the file.
    with open(TEXT_FILE, 'r', encoding='utf8') as file:
        text = file.read()
        source = list(text)

    for i in range(len(source)):
        source[i] = char_to_vec[source[i]]

    # Feed the text to the network.
    # Here we look at the activation of the neurons of the
    # hidden state at the 2nd LSTM layer.
    # We take the first element of the output as there is only one
    # batch.
    out = second_layer.forward(first_layer.forward(np.array([source])))[0]

    # ###############---TKINTER---#############################################
    class Wrap:
        NEURON_INDEX = 0

    def showNeuron():
        for j in range(out.shape[0]):
            # We will leave the background of the newline characters white,
            # regardless of its activation. The reason for that is that the color
            # would fill the entire remainder of the line, which is very disturbing to look at.
            intensity = 255 if text[j] == '\n' else 255 - int((out[j, Wrap.NEURON_INDEX, 0] + 1) * 127.5)
            text_box.tag_config(str(j), background="#%02x%02x%02x" % (
                255, intensity, intensity))

    def inputFromEntry(evt):
        Wrap.NEURON_INDEX = int(entry.get())
        entry.delete(0, "end")
        showNeuron()

    def nextButtonClicked():
        Wrap.NEURON_INDEX += 1
        entry.delete(0, "end")
        entry.insert(tk.INSERT, str(Wrap.NEURON_INDEX))
        showNeuron()

    # Making the tkinter window.
    root = tk.Tk()
    text_box = tk.Text(root, height=35)
    text_box.insert(tk.INSERT, text)
    text_box.pack()
    current_line = 1
    current_char = 0
    for i in range(out.shape[0]):
        text_box.tag_add(str(i), f"{current_line}.{current_char}")
        current_char += 1
        if text[i] == '\n':
            current_line += 1
            current_char = 0

    # Making the entry box.
    entry = tk.Entry(root, width=5)
    entry.pack()
    entry.bind("<Return>", inputFromEntry)

    # Buttons
    up = tk.Button(text="Next", command=nextButtonClicked)
    up.pack()

    # Show the first neuron by default.
    showNeuron()

    root.mainloop()


if __name__ == '__main__':
    main()
