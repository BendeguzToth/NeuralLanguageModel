"""
In this file losses are defined.
"""

# Standard libraries
from abc import ABC, abstractmethod

# Third-party libraries
import numpy as np


class BaseLossFunction(ABC):
    """
    Base class for all the loss functions used with the neural network.
    Override this class to create loss functions.
    """
    @staticmethod
    @abstractmethod
    def error(result, label):
        """
        Override this function to return the error term.
        :param result: The activation of the output layer.
        :type result: np.array with shape=(n, 1)
        :param label: The desired output of the output layer.
        :type label: np.array with shape=(n, 1)
        :return: The (vectorized) error of the network.
        """

    @staticmethod
    @abstractmethod
    def gradient(result, label):
        """
        Override this function to return the derivative of the
        error.
        :param result: The activation of the output layer.
        :type result: np.array with shape=(n, 1)
        :param label: The desired output of the output layer.
        :type label: np.array with shape=(n, 1)
        :return: The (vectorized) gradient of the loss.
        """


class MSE(BaseLossFunction):
    """
    This class implements the mean squared error.
    """
    @staticmethod
    def error(result, label):
        return (1/2) * (label - result) ** 2

    @staticmethod
    def gradient(result, label):
        return -(label - result)


class CrossEntropy(BaseLossFunction):
    """
    This class implements element-wise cross-entropy loss function on the output layer.
    The output neurons are required to have an activation, a, such that 0 <= a <= 1
    ===================================================================================
    Use this with sigmoid output layer!
    """
    @staticmethod
    def error(result, label):
        return - (label * np.log(result) + (1 - label) * np.log((1 - result + 1e-7)))

    @staticmethod
    def gradient(result, label):
        return - (label - result) / (result * (-result + 1) + 1e-7)


class VectorCrossEntropy(BaseLossFunction):
    """
    This class implements a vectorized version of the cross entropy loss.
    To use this loss function, the sum of the activations of the output layer needs to be 1.
    ========================================================================================
    Use this with softmax output layer!
    """
    @staticmethod
    def error(result, label):
        return - label * np.log(result + 1e-7)

    @staticmethod
    def gradient(result, label):
        return - label / (result + 1e-7)
