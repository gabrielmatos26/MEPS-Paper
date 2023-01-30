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

from __future__ import print_function
from neat.math_util import mean
from utils.sort_population_mo_simple import select_n_best, select_n_best_hvc
from neat.reproduction import DefaultReproduction

import numpy as np
import copy

class Population(object):
    def __init__(self, config, fitness_function, initial_state=None, **fitness_kwargs):
        """
        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.
        """
        self.config = config
        self.reproduction = DefaultReproduction()
        self.fitness_function = fitness_function

        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)

            self.fitness_function(list(self.population.items()), self.config, **fitness_kwargs)
            self.generation = 0
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genomes = None
        self.best_genome = None
        self.memory = None
        self.trajectory_archive = None


    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def reset_population(self, config, previous_genomes):
        self.population = self.reproduction.create_new_from_previous(config.genome_type,
                                                                    config.genome_config,
                                                                    config.pop_size,
                                                                    previous_genomes)
        self.best_genomes = None
        self.best_genome = None
        self.memory = None
        self.trajectory_archive = None

    def run_once(self, max_gen, max_solutions=50, adpt_scale=0.0, alpha=1.0, use_hvc=True, **fitness_kwargs):
        self.fitness_function(list(self.population.items()), self.config, **fitness_kwargs)
        
        self.population, current_parents, current_offspring = \
            self.reproduction.reproduce(self.config, self.config.pop_size, self.fitness_function, self.population, self.generation, max_gen, \
                                        adpt_scale, alpha, use_hvc, **fitness_kwargs)

        if self.memory is None:
            self.memory = dict()

        for _, solution in self.population.items():
            fit = tuple(solution.fitness)
            if fit not in self.memory:
                self.memory[fit] = solution
            
        fit_keys = np.array(list(map(list, self.memory.keys())))
        if not use_hvc:
            n_best_keys = select_n_best(fit_keys, max_solutions, only_nondominated=True)
        else:
            n_best_keys = select_n_best_hvc(fit_keys, max_solutions, only_nondominated=True)
        
        new_memory_best = {}
        
        for k in n_best_keys:
            new_memory_best[tuple(k)] = copy.deepcopy(self.memory[tuple(k)])

        self.memory = dict(new_memory_best)

        self.generation += 1

        return self.population, current_parents, current_offspring