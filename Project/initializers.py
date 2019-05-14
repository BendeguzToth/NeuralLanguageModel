"""
In this file we define initializers, the classes
that can be used to initialize various parameters of
networks, such as weights, biases and hidden states.
"""

# Standard libraries
from abc import ABC, abstractmethod

# Third-party libraries
import numpy as np


class Initializer(ABC):
    """
    Abstract base class for initializers.
    """
    @staticmethod
    @abstractmethod
    def initialize(shape):
        """
        This function returns a randomly initialized
        parameter with the given shape.
        :param shape: Shape of the tensor.
        :return: Randomly initialized tensor of shape
        'shape'.
        """


class StandardGaussian(Initializer):
    """
    This class implements standard gaussian
    initialization.
    """
    @staticmethod
    def initialize(shape):
        return np.random.randn(shape[0], shape[1])


class Xavier(Initializer):
    """
    This class implements the Xavier
    initialization method.
    """
    @staticmethod
    def initialize(shape):
        return np.random.randn(shape[0], shape[1]) / np.sqrt((shape[0]))


class XavierReLU(Initializer):
    """
    This class implements the Xavier
    initialization method for ReLUs.
    """
    @staticmethod
    def initialize(shape):
        return np.random.randn(shape[0], shape[1]) / np.sqrt((shape[0] / 2))


class NullInitializer(Initializer):
    """
    This class implements null initialization.
    """
    @staticmethod
    def initialize(shape):
        return np.zeros(shape, dtype="float64")
