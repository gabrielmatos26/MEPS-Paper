###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################


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