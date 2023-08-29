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

from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config, **kwargs):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config), kwargs))

        # assign the fitness back to each genome
        # output_genomes = []
        for job, (gid, genome) in zip(jobs, genomes):           
            modified_genome = job.get(timeout=self.timeout)
            genome.fitness = modified_genome.fitness
            genome.trajectory = modified_genome.trajectory
            genome.q_values = modified_genome.q_values
            genome.a_trajectory = modified_genome.a_trajectory
            genome.preference = modified_genome.preference

    
    def evaluate_novelty_search(self, genomes, distances, config, **kwargs):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, genomes, distances, config), kwargs))

        # assign the fitness back to each genome
        for job, (gid, genome) in zip(jobs, genomes):           
            modified_genome = job.get(timeout=self.timeout)
            genome.fitness = modified_genome.fitness
            genome.trajectory = modified_genome.trajectory
            genome.a_trajectory = modified_genome.a_trajectory
