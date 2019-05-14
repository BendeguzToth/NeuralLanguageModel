"""
In this file activation functions are defined to be
used with the neural network.
"""

# Standard libraries
from abc import ABC, abstractmethod

# Third-party libraries
import numpy as np


class Activation(ABC):
    """
    Base class for activation functions.
    """
    def __init__(self):
        self.x = None

    @abstractmethod
    def activation(self, x):
        """
        This function evaluates the activation on
        the given input.
        :param x: The input to the activation.
        :return: The input after the activation applied.
        """

    @abstractmethod
    def gradient(self, gradient_from_above):
        """
        This function calculates the gradient
        through the activation.
        :param gradient_from_above: The gradient flowing in.
        :return: The gradient of the input of the function
        times the inflowing gradient (and therefore the derivative
        wrt the overall loss.)
        """


class ReLU(Activation):
    """
    This class implements the ReLU activation function.
    """
    def __init__(self, leak=0.):
        super(ReLU, self).__init__()
        self.leak = leak

    def activation(self, x):
        self.x = x
        return np.maximum(x, self.leak * x)

    def gradient(self, gradient_from_above):
        return np.where(self.x < 0, self.leak, 1) * gradient_from_above


class Sigmoid(Activation):
    """
    This class implements the Sigmoid activation function.
    """
    def activation(self, x):
        self.x = x
        return 1 / (1 + np.e ** -x)

    def gradient(self, gradient_from_above):
        return gradient_from_above * np.exp(-self.x) / (1 + np.exp(-self.x)) ** 2


class Tanh(Activation):
    """
    This class implements the Tanh actiovation function.
    """
    def activation(self, x):
        self.x = x
        return np.tanh(x)

    def gradient(self, gradient_from_above):
        return (1 - (np.tanh(self.x) ** 2)) * gradient_from_above


class DenseSoftmax(Activation):
    """
    This class implements the softmax activation, with dense
    inflowing gradients. If the gradient of the loss with
    respect to this the layer is sparse (has only one non-zero
    element) consider using SparseSoftmax instead!
    """
    def activation(self, x):
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        S = np.sum(exps, axis=1, keepdims=True)
        self.x = exps / S
        return self.x

    def gradient(self, gradient_from_above):
        """
        Returns the gradient of a whole batch.
        :param gradient_from_above: 2D array of gradients.
        :return:
        """
        local_grad = np.matmul(-self.x, np.transpose(self.x, axes=(0, 2, 1))) * np.repeat(np.expand_dims(1 - np.identity(self.x.shape[1]), axis=0), self.x.shape[0], axis=0) + (1 - self.x) * self.x * np.repeat(np.expand_dims(np.identity(self.x.shape[1]), axis=0), self.x.shape[0], axis=0)
        return np.matmul(local_grad, gradient_from_above)


class SparseSoftmax(Activation):
    """
    This class assumes a sparse inflowing gradient, with
    a single non-zero value.
    Only use with the vectorized version of cross-entropy
    and one-hot labels!
    Operates on batches of data.
    """
    def __init__(self, temperature=1.):
        Activation.__init__(self)
        self.temperature = temperature

    def activation(self, x):
        x /= self.temperature
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        S = np.sum(exps, axis=1, keepdims=True)
        self.x = exps / S
        return self.x

    def gradient(self, gradient_from_above):
        idxs = np.argmax(gradient_from_above != 0, axis=1).flatten()
        out = -self.x * np.expand_dims(self.x[np.arange(len(gradient_from_above)), idxs, :], axis=2)
        out[np.arange(len(gradient_from_above)), idxs, :] = np.sum(self.x * np.expand_dims(self.x[np.arange(len(gradient_from_above)), idxs, :], axis=2), axis=1) + out[np.arange(len(gradient_from_above)), idxs, :]
        return out * np.expand_dims(gradient_from_above[np.arange(len(gradient_from_above)), idxs, :], axis=2) * (1 / self.temperature)


class Identity(Activation):
    """
    This class impelements the identity activation
    function f(x) = x.
    """
    def activation(self, x):
        return x

    def gradient(self, gradient_from_above):
        return gradient_from_above
