###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################


from __future__ import division
import math
import types


def sigmoid_activation(z):
    return z * (1.0 - z)

def tanh_activation(z):
    return 1.0 - z**2

def relu_activation(z):
    return 1.0 if z > 0.0 else 0.0
    

class InvalidActivationFunction(TypeError):
    pass

class ActivationDerivativesFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.functions['sigmoid'] = sigmoid_activation
        self.functions['tanh'] = tanh_activation
        self.functions['relu'] = relu_activation

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No derivative for such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
