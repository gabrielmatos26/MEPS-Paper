###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################
# Copyright (c) 2023, Gabriel Matos Leite
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
