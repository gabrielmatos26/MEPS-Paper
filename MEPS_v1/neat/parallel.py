###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################


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
