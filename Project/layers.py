"""
In this file the different type of layers
are defined, that can be used to build up
a network.
"""

# Standard libraries
from abc import ABC, abstractmethod
from collections import deque

# Third-party libraries
import numpy as np

# Project files
from optimizers import Optimizer
from initializers import Initializer, Xavier, NullInitializer
from activations import Activation


class Layer(ABC):
    """
    This class defines a simple interface for the layer classes.
    """
    @abstractmethod
    def forward(self, batch):
        """
        The forward pass of the network.
        :param batch: A batch of data.
        :return: The activation of the layer.
        """

    @abstractmethod
    def backward(self, gradient):
        """
        The backward flow of the layer. Updates the
        parameters of the layer.
        :param gradient: The gradient of the loss function
        with respect to the activation of this layer.
        :return: The gradient of the loss function with
        respect to the inputs of this layer.
        """


class Trainable:
    """
    This is a base class for trainable structures.
    (Layers, or filters in a conv net.)
    """
    def __init__(self):
        self.optimizer = None

    def assignOptimizer(self, optimizer):
        """
        Assigning the optimizer to the layer.
        :param optimizer: The optimizer object that
        will be used for this layer. Needs to inherit
        from 'Optimizer'.
        """
        assert issubclass(type(optimizer), Optimizer), "Argument 'optimizer' of Layer.assignOptimizer(optimizer) needs to be an instance of a subclass of Optimizer."
        self.optimizer = optimizer
        self.optimizer.assignLayer(self)

    @abstractmethod
    def saveParams(self):
        """
        This function returns the parameters of the network
        as a json serializable object.
        :return: A serializable object.
        """

    @abstractmethod
    def loadParams(self, data):
        """
        This function loads the parameters from the given json
        object. Inverse of saveParams().
        :param data: Json object.
        """


class MemoryUnit(ABC):
    """
    This is a base class for layer that have a
    way to preserve data across multiple steps.
    """
    @abstractmethod
    def reset(self):
        """
        This function resets the internal state of
        the layer.
        """


class Dense(Layer, Trainable):
    """
    This class implements fully connected layers.
    """
    def __init__(self, size, input_size, activation, initializer=Xavier):
        Trainable.__init__(self)

        assert issubclass(type(activation), Activation), "Argument 'activation' of Dense.__init__(size, input_size, activation, initializer) needs to be an instance of a subclass of 'Activation'"

        self.w = initializer.initialize((size, input_size))
        self.b = initializer.initialize((size, 1))

        self.activation = activation

        self.input_sum = None
        self.input = None

    def forward(self, batch):
        self.input = batch
        self.input_sum = np.matmul(self.w, batch) + self.b
        return self.activation.activation(self.input_sum)

    def backward(self, gradient):
        d_input_sum = self.activation.gradient(gradient)
        self.optimizer.optimize(self, [self.w, self.b], [np.average(np.matmul(d_input_sum, np.transpose(self.input, axes=(0, 2, 1))), axis=0), np.average(d_input_sum, axis=0)])

        return np.matmul(self.w.T, d_input_sum)

    def saveParams(self):
        return [self.w.tolist(), self.b.tolist()]

    def loadParams(self, data):
        self.w = np.array(data[0], dtype="float64")
        self.b = np.array(data[1], dtype="float64")


class TimeDistributed(Layer, Trainable, MemoryUnit):
    """
    This is a wrapper layer for non-recurrent layers, that come after
    recurrent layers in the architecture. The output of a recurrent
    layer has 4 dimensions: BxSxFx1, where B is batch size, S is sequence
    length, F is feature length of the data at a time step, and 1 because the
    features are column vectors. The non-recurrent layers however expect a
    3 dimensional input, with axis BxFx1 (no sequence).
    This wrapper class takes 4D arguments from recurrent layers, and transform
    them to 3D, by merging the first two dimensions (batch and sequence) together.
    Then it will pass it forward through all the layers that has been wrapped into
    it, reshape the final output, to have the original sizes along the first 2 axis.
    """
    def __init__(self, *args):
        """
        Ctor.
        :param args: List of non-recurrent layers, that are
        expected to get a 4D input.
        """
        Trainable.__init__(self)

        self.layers = list(args)
        self.needs_reset = []
        self.trainables = []

        for layer in self.layers:
            assert issubclass(type(layer), Layer), "Argument 'layer' of Network.__init__(*layers) needs to onlny contain instances of a subclass of 'Layer'."
            if issubclass(type(layer), MemoryUnit):
                self.needs_reset.append(layer)
            if issubclass(type(layer), Trainable):
                self.trainables.append(layer)

    def reset(self):
        for layer in self.needs_reset:
            layer.reset()

    def forward(self, batch):
        shape = batch.shape
        batch = np.reshape(batch, newshape=(shape[0] * shape[1], shape[2], shape[3]))

        for layer in self.layers:
            batch = layer.forward(batch)

        return np.reshape(batch, newshape=(shape[0], shape[1], batch.shape[1], batch.shape[2]))

    def backward(self, gradient):
        shape = gradient.shape
        gradient = np.reshape(gradient, newshape=(shape[0] * shape[1], shape[2], shape[3]))

        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].backward(gradient)

        return np.reshape(gradient, newshape=(shape[0], shape[1], gradient.shape[1], gradient.shape[2]))

    def assignOptimizer(self, optimizer):
        for layer in self.trainables:
            layer.assignOptimizer(optimizer)

    def saveParams(self):
        params = []
        for layer in self.trainables:
            params.append(layer.saveParams())
        return params

    def loadParams(self, data):
        counter = 0
        for layer in self.trainables:
            layer.loadParams(data[counter])
            counter += 1


class LSTM(Layer, Trainable, MemoryUnit):
    """
    This class implements the LSTM layer. It returns its
    hidden state at every time step.
    """
    def __init__(self, size, input_size, batch_size, backprop_depth=50, weight_initializer=Xavier, bias_initializer=NullInitializer,
                 hidden_state_initializer=NullInitializer, cell_state_initializer=NullInitializer, forget_bias_to_one=True, stateful=False):
        """
        Ctor.
        :param size: The number of hidden units in the layer.
        :param input_size: The length of the previous layer.
        :param batch_size: The size of a batch that
        will be trained parallel. First dimension of the input
        tensor. It can be changed later, but the network needs
        to be reset to apply the changes.
        :param backprop_depth: k2 parameter of TBPTT. The number of
        time steps that backpropagation will happen. For each time step
        a huge amount of data needs to be kept in memory for backpropagation,
        setting this value to a higher number will increase memory usage.
        :param weight_initializer: The initializer that will be used
        for the weight matrix. Needs to be an instance of a subclass
        of 'Initializer'.
        :param bias_initializer: The initializer that will be used
        for the bias vector. Needs to be an instance of a subclass
        of 'Initializer'.
        :param hidden_state_initializer: The initializer used for
        the hidden state.
        :param cell_state_initializer: The initializer used for
        the cell state.
        :param forget_bias_to_one: If True, the forget gate biases
        are initialized to ones.
        :param stateful: If False, the hidden states will be reset after every
        backward pass. If not, hidden states are preserved, as long as the batch
        dimensions do not change. Hidden states are saved per data point
        in the batch, will have dimensions BxHx1, where B is batch size,
        H is size of the hidden layer, and 1 is because it is a column vector.
        """
        Trainable.__init__(self)

        assert issubclass(weight_initializer, Initializer), "Parameter 'weight_initializer' of LSTM needs to be a subclass of Initializer."
        assert issubclass(bias_initializer, Initializer), "Parameter 'bias_initializer' of LSTM needs to be a subclass of Initializer."
        assert issubclass(hidden_state_initializer, Initializer), "Parameter 'hidden_state_initializer' of LSTM needs to be a subclass of Initializer."
        assert issubclass(cell_state_initializer, Initializer), "Parameter 'cell_state_initializer' of LSTM needs to be a subclass of Initializer."

        self.size = size
        self.k2 = backprop_depth
        self.stateful = stateful

        # The single weight parameter matrix, that is a concatenation of
        # of the 8 sub-matrices:
        # [[Wix  Wih]
        # [Wfx  Wfh]
        # [Wox  Woh]
        # [Wgx  Wgh]]
        # Where x, h are the input at t, and hidden state at t-1,
        # i, f, o, g are input, forget, output, g gates.
        self.w = weight_initializer.initialize((4 * size, input_size + size))

        # The single bias vector, a concatenation of the following:
        # [[Bi]
        # [Bf]
        # [Bo]
        # [Bg]]
        # Where: i, f, o, g -> input, forget, output, g.
        self.b = bias_initializer.initialize((4 * size, 1))
        if forget_bias_to_one:
            self.b[size:2 * size, :].fill(1.)

        # Initializing the hidden states. These will be set in
        # 'onStartTraining', because there is where we know the
        # sequence length (for the cache).
        self.h = None
        self.c = None

        self.inputs = None
        self.input_ifog = None
        self.ifog = None

        self.batch_size = batch_size
        self.hidden_state_initializer = hidden_state_initializer
        self.cell_state_initializer = cell_state_initializer

        self.reset()

    def reset(self):
        self.h = deque(maxlen=self.k2 + 1)
        self.c = deque(maxlen=self.k2 + 1)
        self.inputs = deque(maxlen=self.k2)
        self.input_ifog = deque(maxlen=self.k2)
        self.ifog = deque(maxlen=self.k2)

        # Initializing the hidden states
        self.h.append(self.hidden_state_initializer.initialize((self.batch_size, self.size, 1)))
        self.c.append(self.cell_state_initializer.initialize((self.batch_size, self.size, 1)))

    def forward(self, batch):
        """
        The forward pass of the network, with a batch of
        sequences with dimensions BxSxFx1, where B is batch
        size, S is sequence length, F is feature size.
        :param batch: 4D numpy array.
        :return: 4D numpy array of result, the first 2 dimensions
        matching with the input.
        """
        out = []
        # We need to transpose here in order to switch the first dimension
        # from the batch index to the time index, So in the for loop we
        # get the time step data across the whole batch.
        # The incoming data has the dimensions BxSxFx1 where B is batch axis,
        # S is time step axis, F is feature length axis, and 1 is because the
        # feature is a column vector.
        # But we want to iterate along the S axis, so we need to flip the
        # order of the fist two dimensions. W then get SxBxFx1
        for x in np.transpose(batch, axes=(1, 0, 2, 3)):
            # Caching for backprop.
            self.inputs.append(x)
            # The input sum of the activation of the i, f, o, g
            # gates.
            # sec = self.h[-1]
            d = np.concatenate((x, self.h[-1]), axis=1)
            self.input_ifog.append(np.matmul(self.w, np.concatenate((x, self.h[-1]), axis=1)) + self.b)

            # Putting everyone through the activations.
            ifo = self.sigmoid(self.input_ifog[-1][:, :-self.size, :])
            g = np.tanh(self.input_ifog[-1][:, -self.size:, :])

            # Caching for backprop.
            self.ifog.append(np.concatenate((ifo, g), axis=1))

            # ct = f * ct-1 + i * g
            self.c.append(ifo[:, self.size: 2 * self.size, :] * self.c[-1] + ifo[:, :self.size, :] * g)

            # ht = o * tanh(ct)
            self.h.append(ifo[:, -self.size:, :] * np.tanh(self.c[-1]))

            out.append(self.h[-1])

        # Now we need to flip back the dimensions, so that it is BxSxFx1 again.
        return np.transpose(np.array(out), axes=(1, 0, 2, 3))

    def backward(self, gradient):
        """
        Implements batch TBPTT, with parameter update.
        :param gradient: Tensor with dimensions BxSxFx1.
        :return: Gradient of the activation of the previous layer,
        across batch and time.
        """
        # The derivative of T+1 is just zero, since it has
        # never been executed. However when doing backprop, we
        # assume a gradient signal flowing in from the top for
        # both state c and h, so the implementation nicely fits
        # into a for loop. To avoid the problem of non-existing
        # gradient values, we simply initialize them to zeros.
        d_h = np.zeros((self.batch_size, self.size, 1))
        d_c_t = np.zeros((self.batch_size, self.size, 1))

        # Here we store the sum of the gradients
        # of our trainable parameters across the sequence.
        cumulative_b = 0
        cumulative_w = 0

        # The list of the gradients of the input values.
        # This list will be returned.
        gradients_out = []

        # Transpose gradients to have the time step axis first.
        gradients = np.transpose(gradient, axes=(1, 0, 2, 3))

        # Looping through the whole cache from backwards.
        for t in range(-1, -len(self.inputs) - 1, -1):
            # The gradient at the hidden state.
            d_h = gradients[t] + d_h

            # The gradient at 'o'.
            d_o = d_h * np.tanh(self.c[t]) * self.dsigmoid(self.input_ifog[t][:, 2 * self.size: 3 * self.size, :])

            # The gradient at ct.
            d_c_t = d_h * self.ifog[t][:, 2 * self.size:3 * self.size, :] * self.dtanh(self.c[t]) + d_c_t

            # The gradient at f.
            d_f = d_c_t * self.c[t - 1] * self.dsigmoid(self.input_ifog[t][:, self.size:2 * self.size, :])

            # The gradient at i,
            d_i = d_c_t * self.ifog[t][:, 3 * self.size:, :] * self.dsigmoid(self.input_ifog[t][:, :self.size, :])

            # The gradient at g.
            d_g = d_c_t * self.ifog[t][:, :self.size, :] * self.dtanh(self.input_ifog[t][:, 3 * self.size:, :])

            # The gradient at ct-1
            d_c_t = d_c_t * self.ifog[t][:, self.size:2 * self.size, :]

            # Now we concatenate the gradients, so that we can backprop
            # them in a single matrix operation, instead of 4.
            d_ifog = np.concatenate((d_i, d_f, d_o, d_g), axis=1)

            # Backpropagating the bias. It is the same as the gradient
            # before the activation.
            cumulative_b += d_ifog

            # Backpropagating into the weight matrix.
            cumulative_w += np.matmul(d_ifog, np.transpose(np.concatenate((self.inputs[t], self.h[t - 1]), axis=1),
                                                           axes=(0, 2, 1)))
            # cumulative_w += d_ifog.dot(np.concatenate((self.inputs[t], self.h[t-1]), axis=0).T)

            # The derivative matrix of the inputs and the hidden state of the previous
            # time step.
            d_x_h = np.matmul(self.w.T, d_ifog)
            # d_x_h = self.w.T.dot(d_ifog)

            # The derivative of the input value is the 1st part of the matrix.
            # We insert it to the left side, since we are looping through the
            # samples backwards.
            gradients_out.insert(0, d_x_h[:, :self.size, :])

            # The derivative of ht-1 is the 2nd part of the matrix.
            d_h = d_x_h[:, -self.size:, :]

        # Update the parameters
        self.optimizer.optimize(self, [self.w, self.b], [np.average(cumulative_w, axis=0), np.average(cumulative_b, axis=0)])

        # Reset if not stateful.
        if not self.stateful:
            self.reset()

        # We transpose the array back to having the batch axis first again.
        return np.transpose(np.array(gradients_out), axes=(1, 0, 2, 3))

    def saveParams(self):
        """
        This does not retain the hidden states!
        """
        return [self.w.tolist(), self.b.tolist()]

    def loadParams(self, data):
        self.w = np.array(data[0], dtype="float64")
        self.b = np.array(data[1], dtype="float64")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def dsigmoid(x):
        """
        Derivative of sigmoid.
        """
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    @staticmethod
    def dtanh(x):
        """
        The derivative of the tanh function.
        """
        return 1 - np.tanh(x) ** 2
