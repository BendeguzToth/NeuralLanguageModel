"""
In this file the optimizers are defined that can be
used to optimize the parameters of layers.
"""

# Standard libraries
from abc import ABC, abstractmethod
import json

# Third-party libraries
import numpy as np


class Optimizer(ABC):
    """
    This is a base class for optimizers
    used with the recurrent layers.
    """
    @abstractmethod
    def optimize(self, layer, parameters, gradients):
        """
        This function optimizes the given variables
        based on the (accumulated) gradients.
        :param layer: The owner of the parameters.
        :param parameters: List of the parameters,
        same shape as 'gradients'.
        :param gradients: List of gradients, same shape
        as 'parameters'.
        """

    @abstractmethod
    def clear(self):
        """
        This function resets the optimizer after the
        training session is over.
        """

    @abstractmethod
    def assignLayer(self, layer):
        """
        This function gets called when a layer
        subscribes to this optimizer.
        :param layer: The layer object.
        """


class SGD(Optimizer):
    """
    This class implements the gradient
    descent optimizer.
    """
    def __init__(self, learning_rate=lambda n: 1e-2):
        """
        Ctor.
        :param learning_rate: Function, that takes parameter
        'n', the number of update steps.
        """
        self.learning_rate = learning_rate
        self.n = 1

    def assignLayer(self, layer):
        pass

    def optimize(self, layer, parameters, gradients):
        for i in range(len(gradients)):
            parameters[i] -= self.learning_rate(self.n) * gradients[i]
        self.n += 1

    def clear(self):
        self.n = 1


class Momentum(Optimizer):
    """
    This class implements the SGD momentum
    update rule.
    """
    def __init__(self, learning_rate=lambda n: 1e-2, mu=0.9):
        """
        Ctor.
        :param learning_rate: Learning rate.
        :param mu: Momentum decay parameter.
        """
        self.n = 1
        self.learning_rate = learning_rate
        self.mu = mu
        self.v = dict()

    def assignLayer(self, layer):
        self.v[layer] = [0, 0]

    def optimize(self, layer, parameters, gradients):
        self.v[layer][0] = self.mu * self.v[layer][0] - self.learning_rate(self.n) * gradients[0]
        parameters[0] += self.v[layer][0]
        self.v[layer][1] = self.mu * self.v[layer][1] - self.learning_rate(self.n) * gradients[1]
        parameters[1] += self.v[layer][1]

        self.n += 1

    def clear(self):
        self.n = 1
        for key, value in self.v.items():
            self.v[key] = [0, 0]


class NesterovMomentum(Optimizer):
    """
    This class implements Nesterov accelerated gradient.
    """
    def __init__(self, learning_rate=lambda n: 1e-2, mu=0.9):
        """
        Ctor.
        :param learning_rate: Learning rate.
        :param mu: Momentum decay parameter.
        """
        self.n = 1
        self.learning_rate = learning_rate
        self.mu = mu
        self.v = dict()

    def assignLayer(self, layer):
        self.v[layer] = [0, 0]

    def optimize(self, layer, parameters, gradients):
        v_prev = self.v[layer]
        self.v[layer][0] = self.mu * self.v[layer][0] - self.learning_rate(self.n) * gradients[0]
        parameters[0] += -self.mu * v_prev[0] + (1 + self.mu) * self.v[layer][0]
        self.v[layer][1] = self.mu * self.v[layer][1] - self.learning_rate(self.n) * gradients[1]
        parameters[1] += -self.mu * v_prev[1] + (1 + self.mu) * self.v[layer][1]

        self.n += 1

    def clear(self):
        self.n = 1
        for key, value in self.v.items():
            self.v[key] = [0, 0]


class Adagrad(Optimizer):
    """
    This class implements the adagrad update
    method.
    """
    def __init__(self, learning_rate=lambda n:1e-2):
        """
        Ctor.
        :param learning_rate: Lambda function of training step.
        """
        self.cache = dict()
        self.learning_rate = learning_rate

        self.n = 1

    def assignLayer(self, layer):
        self.cache[layer] = [0, 0]

    def optimize(self, layer, parameters, gradients):
        self.cache[layer][0] += gradients[0] ** 2
        self.cache[layer][1] += gradients[1] ** 2

        parameters[0] -= self.learning_rate(self.n) * gradients[0] / (np.sqrt(self.cache[layer][0]) + 1e-7)
        parameters[1] -= self.learning_rate(self.n) * gradients[1] / (np.sqrt(self.cache[layer][1]) + 1e-7)

    def clear(self):
        self.n = 1
        for key, value in self.cache.items():
            self.cache[key] = [0, 0]


class RMSprop(Optimizer):
    """
    This class implements RMS Prop optimmizer.
    """
    def __init__(self, learning_rate=lambda n:1e-3, decay_rate=0.95):
        """
        Ctor.
        :param learning_rate: The learning rate.
        :param decay_rate: The "leaking rate" of the cache.
        At every update, the cache will be multiplied by this value,
        in order to prevent exploding over time.
        """
        self.n = 1
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.cache = dict()

    def assignLayer(self, layer):
        self.cache[layer] = [0, 0]

    def optimize(self, layer, parameters, gradients):
        self.cache[layer][0] = self.decay_rate * self.cache[layer][0] + (1 - self.decay_rate) * gradients[0] ** 2
        self.cache[layer][1] = self.decay_rate * self.cache[layer][1] + (1 - self.decay_rate) * gradients[1] ** 2

        parameters[0] -= self.learning_rate(self.n) * gradients[0] / (np.sqrt(self.cache[layer][0]) + 1e-7)
        parameters[1] -= self.learning_rate(self.n) * gradients[1] / (np.sqrt(self.cache[layer][1]) + 1e-7)

        self.n += 1

    def clear(self):
        self.n = 1
        for key, value in self.cache.items():
            self.cache[key] = [0, 0]

    def save(self, file_):
        obj = {
            'n': self.n,
            'cache': [[value[0].tolist(), value[1].tolist()] for key, value in self.cache.items()]
        }
        with open(file_, 'w') as file:
            json.dump(obj, file)

    def load(self, file_):
        """
        Call after all layer have been assigned!
        """
        with open(file_, 'r') as file:
            obj = json.load(file)
        self.n = obj["n"]
        for i, k_v in zip(range(len(self.cache.items())), self.cache.items()):
            self.cache[k_v[0]] = [np.array(obj["cache"][i][0]), np.array(obj["cache"][i][1])]
