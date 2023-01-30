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

import numpy as np

class BinaryCrowdedTournamentSelection:
    def __init__(self):
        pass

    @staticmethod
    def do(mating_pool):
        if isinstance(mating_pool, dict):
            list_pool = list(mating_pool.items())
        elif isinstance(mating_pool, list):
            list_pool = mating_pool[:]
        else:
            raise TypeError("Mating pool should by a dict or list")

        n = len(list_pool)
        selection = []
        if n == 1:
            key, _ = list_pool[0]
            selection.append(key)
        else:
            for _ in range(n):

                parents = np.random.choice(np.arange(n), size=2, replace=False)
                parent1, parent2 = list_pool[parents[0]], list_pool[parents[1]]
                (key1, candidate1), (key2, candidate2) = parent1, parent2

                winner, loser = None, None
                if candidate1.rank < candidate2.rank:
                    winner = key1
                    loser = key2
                    # selection.append(key1)
                elif candidate1.rank > candidate2.rank:
                    winner = key2
                    loser = key1
                    # selection.append(key2)
                elif candidate1.crowding_distance > candidate2.crowding_distance:
                    winner = key1
                    loser = key2
                    # selection.append(key1)
                elif candidate1.crowding_distance < candidate2.crowding_distance:
                    winner = key2
                    loser = key1
                    # selection.append(key2)
                else: #tie on both criterion leads to random selection
                    if np.random.random() > 0.5:
                        winner = key1
                        loser = key2
                        # selection.append(key1)
                    else:
                        winner = key2
                        loser = key1
                        # selection.append(key2)

                if np.random.random() < 0.0: #0.1: deterministic or probabilistic TODO: create a parameters for this
                    selection.append(loser)
                else:
                    selection.append(winner)

                

        return selection