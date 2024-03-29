###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################


from pareto_qlearning import PQL
from BenchmarkEnvironments.DeepSeaTreasureEnv import DeepSeaTreasureEnv
from tqdm import tqdm, trange
import pygmo as pg
import numpy as np
from datetime import datetime
import pickle

def main():
    # action_policy = "hv"
    action_policy = "pareto"
    env = DeepSeaTreasureEnv()
    gamma = 1
    ref_point = [25, 0]
    # hv_ordering_function = lambda x : [-1*x[1], -1*x[0]]
    hv_ordering_function = lambda x : [-1*x[0], -1*x[1]]
    eps = 0.997
    n_exp = 20
    num_episodes = 40
    n_iterations = 2000
    # n_exp = 2
    # n_iterations = 3

    pbar_exp = tqdm(total=n_exp)
    hv_runs_history = np.zeros((n_exp, n_iterations))
    for exp in range(n_exp):
        pbar_exp.set_description('RUN {0}'.format(exp+1))
        pql = PQL(ref_point, hv_ordering_function)

        pbar = tqdm(total=n_iterations)
        pbar.set_postfix(PO = 0, HV = 0)
        total_solutions = None
        hv_history = np.zeros(n_iterations)
        for i in range(n_iterations):
            pbar.set_description('ITER {0}'.format(i+1))
            # pql.train(num_episodes, action_policy, env, gamma, num_steps=20, eps=eps)
            pql.train(num_episodes, action_policy, env, gamma, eps=eps, n_iteration=i)
            solutions, hv, num_solutions = pql.test(gamma, env)
            # if total_solutions is None:
            #     total_solutions = solutions
            # else:
            #     total_solutions = np.concatenate((total_solutions, solutions))
            hv_history[i] = hv
            pbar.set_postfix(PO = num_solutions, HV = hv)
            pbar.update()
        hv_runs_history[exp] = hv_history
        pbar.close()
        
        # total_solutions = np.unique(total_solutions, axis=0)
        # ndf, _, _, _ = pg.fast_non_dominated_sorting(points = -1*total_solutions)
        # total_solutions = total_solutions[ndf[0]]
        # print(total_solutions)
        print(solutions)

    results = {'hvs': hv_runs_history}
    with open('pql_dst_{0}.pickle'.format(datetime.now().strftime("%d-%m-%Y_%H-%M")), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()