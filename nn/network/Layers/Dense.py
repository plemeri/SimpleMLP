import os

from nn.network.Layers.Layer import *
from utils.utils import *


class Dense(Layer):
    def __init__(self, in_dimension, out_dimension, activation_fn, d_activation_fn, checkpoint_dir=None, scope=""):
        super().__init__(checkpoint_dir, scope)
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.weight = abs(np.random.randn(self.in_dimension, self.out_dimension) * np.sqrt(1/self.in_dimension))
        self.bias = np.zeros(self.out_dimension)
        self.activation_fn = activation_fn
        self.d_activation_fn = d_activation_fn
        if self.checkpoint_dir is not None:
            self.weight_checkpoint_dir = self.checkpoint_dir + '/' + self.scope + '_weight'
            self.bias_checkpoint_dir = self.checkpoint_dir + '/' + self.scope + '_bias'

    def visualize(self):
        print('dense layer -', self.scope, 'in_dimension: ', self.in_dimension, 'out_dimension :', self.out_dimension)

    def load(self):
        self.weight = np.load(self.weight_checkpoint_dir + '.npy')
        self.bias = np.load(self.bias_checkpoint_dir + '.npy')

    def save(self):
        if os.path.isdir(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir)
        np.save(self.weight_checkpoint_dir, self.weight)
        np.save(self.bias_checkpoint_dir, self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        self.a = np.matmul(inputs, self.weight) + self.bias
        self.y = self.activation_fn(self.a)
        return self.y

    def backward(self, gradient_y, learning_rate):
        if len(gradient_y.shape) == 1:
            gradient_y = np.broadcast_to(gradient_y, shape=self.a.T.shape)
            gradient_y = gradient_y.T
        gradient_a = self.d_activation_fn(self.a) * gradient_y
        gradient_w = np.matmul(self.inputs.T, gradient_a)
        gradient_b = np.matmul(np.ones(self.inputs.shape[0]), gradient_a)
        gradient_back = np.matmul(gradient_a, self.weight.T)
        self.weight = self.weight - learning_rate * gradient_w
        self.bias = self.bias - learning_rate * gradient_b
        return gradient_back


