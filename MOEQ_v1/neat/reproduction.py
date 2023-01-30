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

import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from utils.sort_population_mo_simple import *

from selection.crowded_tournament_selection import BinaryCrowdedTournamentSelection
import numpy as np

class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [])

    def __init__(self):
        self.genome_indexer = count(1)
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config, initialization=True)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    def create_new_from_clone(self, genome, genome_type):
        key = next(self.genome_indexer)
        sigma = genome.sigma

        g = genome_type(key, sigma=sigma)

        for ng_key, ng in genome.nodes.items():
            g.nodes[ng_key] = ng.copy()
        
        for cg_key, cg in genome.connections.items():
            g.connections[cg_key] = cg.copy()

        if isinstance(genome.fitness, (int, float)):
            g.fitness = genome.fitness
        elif genome.fitness is not None:
            g.fitness = genome.fitness[:]

        self.ancestors[key] = tuple()
        return key, g

    def generate_offspring(self, config, population, single=False):
        assign_ranks(population, use_hvc=True)

        # perform binary crowded tournament selection
        selection = BinaryCrowdedTournamentSelection.do(population)
        offspring = {}

        for parent_id in selection:
            parent = population[parent_id]
    
            gid, child = self.create_new_from_clone(parent, config.genome_type)
            child.mutate(config.genome_config)
            offspring[gid] = child
            self.ancestors[gid] = (parent_id,parent_id)

        return offspring

    def reproduce(self, config, max_size, fitness_function, population, generation, max_gen, adpt_scale=0.0, alpha=1.0, use_hvc=True, **fitness_kwargs):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        new_population = {}

        candidates = dict(population)
        offspring = self.generate_offspring(config, population)
        fitness_function(list(offspring.items()), config, **fitness_kwargs)

        candidates.update(offspring)

        #update sigma values in offspring
        episode_length = fitness_kwargs['episode_length']
        delta = np.round(0.2 * episode_length).astype(int)
        # delta = np.round(0.3 * episode_length).astype(int)
        if adpt_scale > 0:
            dists = np.zeros(len(offspring))
            for i, (mid, mutated) in enumerate(offspring.items()):
                pid = self.ancestors[mid][0]
                parent = population[pid]
                dist = np.sum(parent.a_trajectory != mutated.a_trajectory)
                dists[i] = dist

            mean_dists = np.mean(dists)
            for gid, g in candidates.items():
                if mean_dists <= delta:
                    g.sigma *= adpt_scale
                else:
                    g.sigma /= adpt_scale
                if fitness_kwargs['extras'] is None:
                    g.sigma = np.clip(g.sigma, 0.2, 2.5) #benchmarks
                else:
                    g.sigma = np.clip(g.sigma, 0.5, 3.5) #microgrid

        assign_ranks(candidates, use_hvc=use_hvc)

        if alpha > 0:
            G = np.round(max_gen/2).astype(int) # set to half max gen
            n_fittest = sort_population_heavy_tail(candidates, max_size, G, R=0.5, Gc=generation, alpha=alpha, use_hvc=use_hvc) #2
        else:
            n_fittest = sort_population(candidates, max_size, use_hvc=use_hvc)

        for key, genome in n_fittest.items():
            new_population[key] = genome
            

        return new_population, dict(population), offspring