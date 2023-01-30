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

import types
import warnings
from functools import reduce
from operator import mul

from neat.math_util import mean, median2


def product_aggregation(x):  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    return max(x)


def min_aggregation(x):
    return min(x)


def maxabs_aggregation(x):
    return max(x, key=abs)


def median_aggregation(x):
    return median2(x)


def mean_aggregation(x):
    return mean(x)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):  # TODO: Recognize when need `reduce`
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction("A function taking at least one argument is required")


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, function):
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
