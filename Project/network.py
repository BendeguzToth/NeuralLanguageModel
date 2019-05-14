# Standard libraries
import json

# Third-party libraries
import numpy as np

# Project files
from layers import Layer, MemoryUnit, Trainable
from losses import BaseLossFunction
from optimizers import Optimizer
from util import IncrementalAverage


class Network:
    """
    This class implements a neural network.
    """
    def __init__(self, *layers):
        """
        Ctor.
        :param layers: Layers of the network.
        """
        self.layers = list(layers)
        self.needs_reset = []
        self.trainables = []

        self.optimizer = None

        for layer in self.layers:
            assert issubclass(type(layer), Layer), "Argument 'layer' of Network.__init__(*layers) needs to onlny contain instances of a subclass of 'Layer'."
            if issubclass(type(layer), MemoryUnit):
                self.needs_reset.append(layer)
            if issubclass(type(layer), Trainable):
                self.trainables.append(layer)

    def addLayer(self, layer):
        """
        This function lets you add a layer to the network.
        :param layer: The layer object ot be added.
        """
        assert issubclass(type(layer), Layer), "Argument 'layer' of Network.addLayer(layer) needs to be an instance of a subclass of 'Layer'."

        self.layers.append(layer)
        if issubclass(type(layer), Trainable):
            self.trainables.append(layer)
        if issubclass(type(layer), MemoryUnit):
            self.needs_reset.append(layer)

    def assignOptimizer(self, optimizer):
        """
        Assigns an optimizer to the layers.
        This function needs to be called before training.
        :param optimizer: The optimizer to be assigned.
        """
        assert issubclass(type(optimizer), Optimizer), "Parameter 'optimizer' of Network.assignOptimizer(optimizer) needs to be an instance of a subclass of 'Optimizer'."

        self.optimizer = optimizer

        for layer in self.trainables:
            layer.assignOptimizer(optimizer)

    def forward(self, batch):
        """
        Forward pass of the network.
        :param batch: Batch of data to evaluate.
        :return: The result.
        """
        for layer in self.layers:
            batch = layer.forward(batch)
        return batch

    def backward(self, gradient):
        """
        Backward pass and parameter update of
        the network.
        :param gradient: The gradient of the loss with respect to
        the activation of the last layer.
        :return: The gradient of the loss with respect to
        the input data.
        """
        for i in range(-1, -len(self.layers) - 1, -1):
            gradient = self.layers[i].backward(gradient)
        return gradient

    def train(self, training_data, lossfunc):
        """
        This function performs 1 epoch train on the provided
        training data.
        :param training_data: List or generator of a 2 tuple (batch, labels) of
        batches. (4D for recurrent, 3D for other layers.)
        :param lossfunc: The loss function to use with training.
        :return The average loss of the epoch.
        """
        assert issubclass(lossfunc, BaseLossFunction), "Parameter 'lossfunc' of Network.train(training_data, lossfunc) needs to be a subclass of 'BaseLossFunction'."

        for layer in self.needs_reset:
            layer.reset()

        loss = IncrementalAverage()
        accuracy = IncrementalAverage()

        for pair in training_data:
            output = self.forward(pair[0])
            accuracy.add(np.count_nonzero(np.argmax(output, axis=-2) == np.argmax(pair[1], axis=-2)) / np.prod(np.array(output.shape[:-2])))
            loss.add(np.average(lossfunc.error(output, pair[1])))
            self.backward(lossfunc.gradient(output, pair[1]))

        return loss.get(), accuracy.get()

    def reset(self):
        """
        This function resets all the layers of the network that have
        an internal state.
        """
        for layer in self.needs_reset:
            layer.reset()

    def saveParams(self, file):
        """
        This function saves the network parameters to the given file.
        Only saves the parameters, not the network architecture!
        :param file: The path to the file to save to.
        """
        data = []
        for layer in self.trainables:
            data.append(layer.saveParams())
        with open(file, 'w') as file:
            json.dump(data, file)

    def loadParams(self, file):
        """
        This function loads the network parameters from the
        given file. The architecture of the network has to
        exactly match the architecture of the network the
        parameters were saved from!
        :param file: Path to the file to load from.
        """
        with open(file, 'r') as file:
            data = json.load(file)
        for layer, params in zip(self.trainables, data):
            layer.loadParams(params)
