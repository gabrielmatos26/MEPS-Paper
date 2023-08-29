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

from math import sqrt, exp


def mean(values):
    values = list(values)
    return sum(map(float, values)) / len(values)


def median(values):
    values = list(values)
    values.sort()
    return values[len(values) // 2]


def median2(values):
    values = list(values)
    n = len(values)
    if n <= 2:
        return mean(values)
    values.sort()
    if (n % 2) == 1:
        return values[n//2]
    i = n//2
    return (values[i - 1] + values[i])/2.0


def variance(values):
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    return sqrt(variance(values))


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..)."""
    e_values = list(map(exp, values))
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2}
