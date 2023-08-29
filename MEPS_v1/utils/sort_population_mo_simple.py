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
from scipy import spatial
import sys
import random
from pymoo.indicators.hv import HV

def cdist(A, B, **kwargs):
    return spatial.distance.cdist(A, B, **kwargs)

def find_duplicates(X, epsilon=1e-16):
    # calculate the distance matrix from each point to another
    D = cdist(X, X)

    # set the diagonal to infinity
    D[np.triu_indices(len(X))] = np.inf

    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(D < epsilon, axis=1)

    return is_duplicate

def dominates(A, B):
    # A and B must be numpy arrays
    assert len(A) == len(B)
    
    if all(A <= B) and any(A < B):
        return 1
    elif all(B <= A) and any(B < A):
        return -1
    return 0

def crowding_distance(front, filter_out_duplicates=True):
    crowding_distances = np.zeros(len(front))
    num_objs = len(front[0])

    if filter_out_duplicates:
        # filter out solutions which are duplicates - duplicates get a zero finally
        is_unique = np.where(np.logical_not(find_duplicates(front, epsilon=1e-24)))[0]
    else:
        # set every point to be unique without checking it
        is_unique = np.arange(len(front))

    filtered_front = np.array(front)[is_unique].tolist()
    enum_front = list(enumerate(filtered_front))
    for obj_idx in range(num_objs):
        enum_front.sort(key=lambda x: x[1][obj_idx])
        crowding_distances[enum_front[0][0]] = sys.maxsize
        crowding_distances[enum_front[-1][0]] = sys.maxsize
        for p in range(1, len(filtered_front) - 1):
            if(crowding_distances[enum_front[p][0]] == sys.maxsize):
                continue
            if (enum_front[-1][1][obj_idx] - enum_front[0][1][obj_idx]) == 0:
                continue
            cd = (enum_front[p+1][1][obj_idx] - enum_front[p-1][1][obj_idx]) / \
                (enum_front[-1][1][obj_idx] - enum_front[0][1][obj_idx])
            crowding_distances[enum_front[p][0]] += cd
    return crowding_distances

def hypervolume_contribution(F, ref=None, return_hv=False):
    """Returns the index of the individual with the least the hypervolume
    contribution. The provided *front* should be a set of non-dominated
    individuals' fitnesses
    """

    if ref is None:
        ref = np.max(F, axis=0) + 1
    
    hv = HV(ref_point=ref)

    def contribution(i):
        # The contribution of point p_i in point set P
        # is the hypervolume of P without p_i
        return hv.do(np.concatenate((np.array(F)[:i,:], np.array(F)[i+1:,:])))

    # Parallelization note: Cannot pickle local function
    contrib_values = [contribution(i) for i in range(len(F))]

    # Select the maximum hypervolume value (correspond to the minimum difference)
    if return_hv:
        return contrib_values
    return np.argmax(contrib_values)

def non_dominated_sorting(F, return_idx=False):
    n_ranks = np.zeros(len(F), dtype=int)
    for i in range(len(F)):
        for j in range(i+1, len(F)):
            dominance = dominates(F[i], F[j])
            if dominance > 0:
                n_ranks[j] += 1
            elif dominance < 0:
                n_ranks[i] += 1
    fronts = [[] for i in range(np.max(n_ranks) + 1)]
    indexes = [[] for i in range(np.max(n_ranks) + 1)]
    for i, rank in enumerate(n_ranks):
        fronts[rank].append(F[i])
        indexes[rank].append(i)
    if return_idx:
        return fronts, indexes
    return fronts
    
def select_n_best(F, n, return_idxs=False, only_nondominated=False):
    if n > len(F):
        n = len(F)
    
    fronts, indexes = non_dominated_sorting(F, True)
    survivor_indexes = []
    survivors = []
    for front_idxs, front in zip(indexes, fronts):
        if len(front) == 0:
            continue
        if len(survivors) >= n:
            break
        if len(front) <= (n - len(survivors)):
            survivors.extend(front)
            survivor_indexes.extend(front_idxs)
        else:
            cds = list(enumerate(crowding_distance(front).tolist()))
            cds.sort(key=lambda x: x[1], reverse=True)
            for i in range(n - len(survivors)):
                survivors.append(front[cds[i][0]])
                survivor_indexes.append(front_idxs[cds[i][0]])
        if only_nondominated:
            break
    if return_idxs:
        return np.array(survivors), np.array(survivor_indexes)
    return np.array(survivors)
                
def select_n_best_hvc(F, n, return_idxs=False, only_nondominated=False):
    if n > len(F):
        n = len(F)
    
    fronts, indexes = non_dominated_sorting(F, True)
    survivor_indexes = []
    survivors = []
    for front_idxs, front in zip(indexes, fronts):
        if len(front) == 0:
            continue
        if len(survivors) >= n:
            break
        if len(front) <= (n - len(survivors)):
            survivors.extend(front)
            survivor_indexes.extend(front_idxs)
        else:
             # Separate the mid front to accept only k individuals
            remaining = n - len(survivors)
            if remaining > 0:
                # reference point is chosen in the complete population
                # as the worst in each dimension +1
                ref = np.max(F, axis=0) + 1

                for _ in range(len(front) - remaining):
                    idx = hypervolume_contribution(front, ref=ref)
                    front.pop(idx)
                    front_idxs.pop(idx)

                
                survivors.extend(front)
                survivor_indexes.extend(front_idxs)
        if only_nondominated:
            break
    if return_idxs:
        return np.array(survivors), np.array(survivor_indexes)
    return np.array(survivors)

def assign_ranks(population, use_hvc=False):
    """
        Sort population of individuals based on pareto dominance and crowd distancing as proposed in NSGA-II and
        assign a rank and crowding distance value to each genome
    """
    # do the non-dominated sorting until splitting front
    F = []
    keys = []
    for key, genome in population.items():
        keys.append(key)
        F.append(list(genome.fitness))
    F = np.array(F)
    keys = np.array(keys)

    # each element in front is a tuple of (genome key, genome position in F list)
    fronts, fronts_idx = non_dominated_sorting(F, True)

    for rank, (front_idx, front) in enumerate(zip(fronts_idx, fronts)):
        if len(front) == 0:
            continue
        
        if use_hvc:
            hv_contribs = hypervolume_contribution(front, return_hv=True)
            
            for hvc, element in zip(hv_contribs, front_idx):
                key = keys[element]
                population[key].rank = rank
                population[key].crowding_distance = 1/hvc if hvc > 0 else 0
        else:
            # calculate the crowding distance of the front
            crowding_of_front = crowding_distance(front)

            # save rank and crowding in each genome
            for cd, element in zip(crowding_of_front,front_idx):
                key = keys[element]
                population[key].rank = rank
                population[key].crowding_distance = cd


def sort_population(population, n, return_indexes=False, return_nondominated_quantity=False, use_hvc=False):
    """
        Sort population of individuals based on pareto dominance and crowd distancing as proposed in NSGA-II.
        Returns the n dominant individuals
    """
    F = []
    keys = []
    for key, genome in population.items():
        keys.append(key)
        F.append(list(genome.fitness))
    F = np.array(F)
    keys = np.array(keys)

    fronts, fronts_idxs = non_dominated_sorting(F, True)

    survivors = []
    survivor_indexes = []

    num_nondominated = len(fronts[0])
    
    for front_idxs, front in zip(fronts_idxs, fronts):
        if len(front) == 0:
            continue
        if len(survivors) >= n:
            break
        if len(front) <= (n - len(survivors)):
            survivors.extend(front)
            survivor_indexes.extend(front_idxs)
        else:
            if use_hvc: #uses hypervolume contribution selection operator       
                # Separate the mid front to accept only k individuals
                remaining = n - len(survivors)
                if remaining > 0:
                    # reference point is chosen in the complete population
                    # as the worst in each dimension +1
                    ref = np.max(F, axis=0) + 1

                    for _ in range(len(front) - remaining):
                        idx = hypervolume_contribution(front, ref=ref)
                        front.pop(idx)
                        front_idxs.pop(idx)
                    
                    survivors.extend(front)
                    survivor_indexes.extend(front_idxs)
            else: #uses crowding distance selection operator
                cds = list(enumerate(crowding_distance(front).tolist()))
                random.shuffle(cds)
                cds.sort(key=lambda x: x[1], reverse=True)
                for i in range(n - len(survivors)):
                    survivors.append(front[cds[i][0]])
                    survivor_indexes.append(front_idxs[cds[i][0]])

    n_fitted_population = {}
    for survivor_idx in survivor_indexes:
        key = keys[survivor_idx]
        n_fitted_population[key] = population[key]
    
    if return_indexes and not return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes)
    elif not return_indexes and return_nondominated_quantity:
        return n_fitted_population, num_nondominated
    elif return_indexes and return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes), num_nondominated
        
    return n_fitted_population

def sort_population_discrete(population, n, return_indexes=False, return_nondominated_quantity=False):
    """
        Sort population of individuals based on pareto dominance and crowd distancing as proposed in NSGA-II.
        Returns the n dominant individuals
    """
    F = []
    keys = []
    for key, genome in population.items():
        keys.append(key)
        F.append(list(genome.fitness))
    F = np.array(F)
    keys = np.array(keys)

    fronts, fronts_idxs = non_dominated_sorting(F, True)

    survivors = []
    survivor_indexes = []

    num_nondominated = len(fronts[0])
    
    for front_idxs, front in zip(fronts_idxs, fronts):
        if len(front) == 0:
            continue
        if len(survivors) >= n:
            break
        
        is_unique = np.where(np.logical_not(find_duplicates(front, epsilon=1e-24)))[0]
        unique_front = front[is_unique]
        unique_front_idxs = front_idxs[is_unique]
        if len(unique_front) <= (n - len(survivors)):
            survivors.extend(unique_front)
            survivor_indexes.extend(unique_front_idxs)
        else:
            cds = list(enumerate(crowding_distance(unique_front).tolist()))
            random.shuffle(cds)
            cds.sort(key=lambda x: x[1], reverse=True)
            for i in range(n - len(survivors)):
                survivors.append(unique_front[cds[i][0]])
                survivor_indexes.append(unique_front_idxs[cds[i][0]])

    n_fitted_population = {}
    for survivor_idx in survivor_indexes:
        key = keys[survivor_idx]
        n_fitted_population[key] = population[key]
    
    if return_indexes and not return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes)
    elif not return_indexes and return_nondominated_quantity:
        return n_fitted_population, num_nondominated
    elif return_indexes and return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes), num_nondominated
        
    return n_fitted_population

def sort_population_heavy_tail(population, n, G, R=0.5, Gc=1, return_indexes=False, return_nondominated_quantity=False, k=1.5, alpha=1.0, use_hvc=False):
    """
        Sort population of individuals based on pareto dominance and crowd distancing as proposed in NSGA-II.
        Returns the n dominant individuals
    """
    
    ratio = R + (1-R)/G * Gc
    if Gc >= G:
        # here, ratio is 1
        return sort_population(population, n, return_indexes=return_indexes, return_nondominated_quantity=return_nondominated_quantity, use_hvc=use_hvc)


    F = []
    keys = []
    for key, genome in population.items():
        keys.append(key)
        F.append(list(genome.fitness))
    F = np.array(F)
    keys = np.array(keys)

    # fronts, fronts_idxs = non_dominated_sorting(F, True)

    xm = 1
    # alpha = alpha
    # alpha = 1.0
    # alpha = (((Gc - 1) / G) * R)
    # alpha = 1 + ((Gc - 1) / G * R)

    n = min(n, len(population))
        

    pareto = lambda x, xm, alpha, N : alpha*xm**alpha / x**(alpha+1) / sum(alpha*xm**alpha / x**(alpha+1)) * N
    survivors = []
    survivor_indexes = []

    # do the non-dominated sorting until splitting front
    fronts, fronts_idxs = non_dominated_sorting(F, True)
    # len_fronts = np.array(list(range(1, len(fronts)+1)))
    # num_per_front = np.round(weibull(len_fronts, l, k, n)).astype(int)
    first_front_num = np.round(ratio * n).astype(int)
    num_per_front = [first_front_num]
    if len(fronts) > 1:
        num_per_front.extend(np.ceil(pareto(np.arange(1, len(fronts)), xm, alpha, n - first_front_num)).astype(int).tolist())

    # scd = special_crowding_distance(population, config)

    while len(survivors) < n:
        num_nondominated = len(fronts[0])
        previous_excess = 0
    
        for k, (front_idxs, front) in enumerate(zip(fronts_idxs, fronts)):
            if len(survivors) >= n:
                break
            if len(front) == 0:
                continue
            available_slots = num_per_front[k] + previous_excess
            # crowding_of_front = np.array([scd[key] for key in front[:,0]])

            if len(front) > available_slots and len(survivors) + available_slots > n:
                if use_hvc:
                    _tmp_front = front[:]
                    # Separate the mid front to accept only k individuals
                    remaining = n - len(survivors)
                    if remaining > 0:
                        # reference point is chosen in the complete population
                        # as the worst in each dimension +1
                        ref = np.max(F, axis=0) + 1
                        hvc_selected = np.arange(len(front)).tolist()
                        for _ in range(len(_tmp_front) - remaining):
                            idx = hypervolume_contribution(_tmp_front, ref=ref)
                            _tmp_front.pop(idx)
                            hvc_selected.pop(idx)
                        
                        I = hvc_selected[:]
                else:
                    # calculate the crowding distance of the front
                    cds = list(enumerate(crowding_distance(front).tolist()))
                    random.shuffle(cds)
                    cds.sort(key=lambda x: x[1], reverse=True)
                    cds = np.array(cds)
                    I = cds[:(n - len(survivors)), 0].astype(int)
            elif len(front) > available_slots:
                if use_hvc:
                    _tmp_front = front[:]
                    # Separate the mid front to accept only k individuals
                    remaining = available_slots
                    if remaining > 0:
                        # reference point is chosen in the complete population
                        # as the worst in each dimension +1
                        ref = np.max(F, axis=0) + 1
                        hvc_selected = np.arange(len(front)).tolist()
                        for _ in range(len(_tmp_front) - remaining):
                            idx = hypervolume_contribution(_tmp_front, ref=ref)
                            _tmp_front.pop(idx)
                            hvc_selected.pop(idx)
                        
                        I = hvc_selected[:]
                else:
                    # calculate the crowding distance of the front
                    cds = list(enumerate(crowding_distance(front).tolist()))
                    random.shuffle(cds)
                    cds.sort(key=lambda x: x[1], reverse=True)
                    cds = np.array(cds)
                    I = cds[:available_slots, 0].astype(int)
            elif len(survivors) + len(front) > n:
                if use_hvc:
                    _tmp_front = front[:]
                    # Separate the mid front to accept only k individuals
                    remaining = n - len(survivors)
                    if remaining > 0:
                        # reference point is chosen in the complete population
                        # as the worst in each dimension +1
                        ref = np.max(F, axis=0) + 1
                        hvc_selected = np.arange(len(front)).tolist()
                        for _ in range(len(_tmp_front) - remaining):
                            idx = hypervolume_contribution(_tmp_front, ref=ref)
                            _tmp_front.pop(idx)
                            hvc_selected.pop(idx)
                        
                        I = hvc_selected[:]
                else:
                    # calculate the crowding distance of the front
                    cds = list(enumerate(crowding_distance(front).tolist()))
                    random.shuffle(cds)
                    cds.sort(key=lambda x: x[1], reverse=True)
                    cds = np.array(cds)
                    I = cds[:(n - len(survivors)), 0].astype(int)
            else:
                previous_excess = available_slots - len(front)
                I = np.arange(len(front)).astype(int)

            # extend the survivors by all or selected individuals
            np_front = np.array(front)
            survivors.extend(np_front[I].tolist())
            survivor_indexes.extend(np.array(front_idxs)[I].tolist())
            fronts[k] = np_front[list(set(np.arange(len(front))) - set(np.arange(len(front))[I]))].tolist()

    n_fitted_population = {}
    for survivor_idx in survivor_indexes:
        key = keys[survivor_idx]
        n_fitted_population[key] = population[key]
    
    if return_indexes and not return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes)
    elif not return_indexes and return_nondominated_quantity:
        return n_fitted_population, num_nondominated
    elif return_indexes and return_nondominated_quantity:
        return n_fitted_population, np.array(survivor_indexes), num_nondominated
        
    return n_fitted_population
